use std::fmt::{Pointer};
use candle_core::{Tensor, Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use anyhow::{Error, Result};
use hf_hub::{Repo};
use hf_hub::api::sync::{Api, ApiRepo};
use candle_core::error::Result as CEResult;
use candle_core::error::Error as CEError;
use candle_core::Module;
use candle_transformers::models::clip::{div_l2_norm, vision_model};
use candle_transformers::models::clip::vision_model::{ClipVisionConfig, ClipVisionTransformer};
use dirs::home_dir;
use reqwest::blocking;


fn set_hf_cache(){
    //     Huggingface uses the path set at env var HF_HOME for managing cache.
    //     Hence the default path is ~/.cache/huggingface/hub
    //     https://github.com/huggingface/hf-hub/blob/9d6502f5bc2e69061c132f523c76a76dad470477/src/lib.rs#L194

    //     Then they make use of the cache before downloading artifacts as seen here:
    //     https://github.com/huggingface/hf-hub/blob/9d6502f5bc2e69061c132f523c76a76dad470477/src/api/sync.rs#L472

    //     Perhaps we should rely on their caching feature from here, but just set it to be in the directory we use for Ahnlich.
    let mut home = home_dir().expect("Could not resolve the home directory.");
    home.push(".cache");
    home.push("ahnlich");
    home.push("ai");
    home.push("huggingface");
    let home_str = home.into_os_string().into_string().unwrap();
    std::env::set_var("HF_HOME", home_str);
}

fn main() -> Result<()>{
    set_hf_cache();
    // Ok(text_main()?)
    Ok(image_main()?)
}

fn image_main() -> Result<()> {
    let (model, image_size) = image_build_model()?;
    let device = Device::Cpu;
    let image_paths = vec![
        "https://www.thesun.co.uk/wp-content/uploads/2024/07/newspress-collage-g74tahww1-1720600843169.jpg".to_string(),
        "https://images.csmonitor.com/csm/2016/09/1003-LRAINER-katwe_1.jpg?alias=standard_900x600nc".to_string(),
        "https://media.wired.com/photos/5926b682cefba457b079ae7c/master/w_1600,c_limit/QK-08053.jpg".to_string(),
    ];
    let images = load_images(&image_paths, image_size)?.to_device(&device)?;
    let tensor = model.forward(&images)?;
    println!("Generated embeddings: \n{}", format!("{}", tensor));
    let sim_scores = tensor.matmul(&tensor.t()?)?;
    println!("Cosine similarities: \n{}", format!("{}", sim_scores));
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ClipVisionOnlyModel {
    model: ClipVisionTransformer,
    visual_projection: candle_nn::Linear,
}

impl ClipVisionOnlyModel {
    pub fn new(vs: candle_nn::VarBuilder, config: &ClipVisionConfig) -> candle_core::Result<Self> {
        let model = ClipVisionTransformer::new(vs.pp("vision_model"), &config)?;
        let visual_projection = candle_nn::linear_no_bias(
            config.embed_dim,
            config.projection_dim,
            vs.pp("visual_projection"),
        )?;

        Ok(Self {
            model,
            visual_projection,
        })
    }

    pub fn get_features(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        // Check the visual projection? Is it right?
        pixel_values
            .apply(&self.model)?
            .apply(&self.visual_projection)
    }

    pub fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        let image_features = self.get_features(pixel_values)?;
        let image_features_normalized = div_l2_norm(&image_features)?;
        Ok(image_features_normalized)
    }
}


fn image_build_model() -> Result<(ClipVisionOnlyModel, usize)>{
    let api = hf_hub::api::sync::Api::new()?;
    let api: ApiRepo = api.repo(hf_hub::Repo::with_revision(
        "openai/clip-vit-base-patch32".to_string(), hf_hub::RepoType::Model,
        "refs/pr/15".to_string()
    ));
    let model_file = api.get("pytorch_model.bin")?;
    let config = vision_model::ClipVisionConfig::vit_base_patch32();
    let varbuilder = VarBuilder::from_pth(&model_file, DTYPE, &Device::Cpu)?;
    let model = ClipVisionOnlyModel::new(varbuilder, &config)?;
    Ok((model, config.image_size))
}


fn text_main() -> Result<()> {
    let (model, mut tokenizer) = text_build_model_and_tokenizer()?;
    let device = &model.device;

    let pp = PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    };

    let text = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ];
    tokenizer.with_padding(Some(pp));

    let tokens = tokenizer.encode_batch(text.to_vec(), true).map_err(Error::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        }).collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let embeddings = mean_pooling(
        &(model.forward(&token_ids, &token_type_ids)?),
        true
    )?;
    println!("generated embeddings \n{}", embeddings);
    let sim_scores = embeddings.matmul(&embeddings.t()?)?;
    println!("Cosine similarities: \n{}", format!("{}", sim_scores));
    Ok(())
}

fn text_build_model_and_tokenizer() -> Result<(BertModel, Tokenizer)>{
    let huggingface_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let repo = Repo::model(huggingface_id);

    // # Set a custom cache path
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("pytorch_model.bin")?;

        (config, tokenizer, weights)
    };

    let config = std::fs::read_to_string(config_filename)?;
    let config : Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
    let varbuilder = VarBuilder::from_pth(&weights_filename, DTYPE, &Device::Cpu)?;
    let model = BertModel::load(varbuilder, &config)?;
    Ok((model, tokenizer))
}

pub fn mean_pooling(v: &Tensor, normalize: bool) -> Result<Tensor> {
    let mean_v = v.mean(1)?;
    if normalize {
        Ok(mean_v.broadcast_div(&mean_v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    } else {
        Ok(mean_v)
    }
}

pub fn load_image<P: AsRef<std::path::Path>>(p: P, res: usize) -> CEResult<Tensor> {
    let mut img;
    let p = p.as_ref();
    if p.starts_with("https://") {
        let bytes = blocking::get(p.to_str().unwrap()).unwrap().bytes().unwrap();
        img = image::load_from_memory(&bytes).unwrap();
    } else {
        img = image::io::Reader::open(p)?.decode().map_err(CEError::wrap)?;
    }
    img = img.resize_to_fill(res as u32, res as u32, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (res as usize, res as usize, 3), &Device::Cpu)?
        .permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

fn load_images<T: AsRef<std::path::Path>>(
    paths: &Vec<T>,
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];

    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor);
    }

    let images = Tensor::stack(&images, 0)?;

    Ok(images)
}
