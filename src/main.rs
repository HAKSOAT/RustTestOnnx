// This file was written using the following as inspiration:
// https://github.com/pykeio/ort/blob/main/examples/gpt2/examples/gpt2.rs

// The ONNX Model and Tokenizer were exported from this notebook
// https://colab.research.google.com/drive/1f5PvBd9kfBs0FCwW9KL-jAuOArXxtf3a?usp=sharing

// They can be accessed from this drive
// https://drive.google.com/file/d/1ErXOHak6uRdc5VTqE-I_G_6dr9D1VUiO/view?usp=sharing
// They should be downloaded into src/gpt2/data

use std::path::Path;
use ndarray::{Array1, Axis, Array, Array2, Ix2};
use tokenizers::Tokenizer;
use ort::{inputs, GraphOptimizationLevel, Session};

const PROJ_DIR: &str = "/Users/haks/Documents/dev/testonnx/src/gpt2";

struct Inference {
    session: Session,
    tokenizer: Tokenizer
}

fn setup_inference() -> Inference {
    ort::init().with_name("Embedder")
        .with_execution_providers([])
        .commit().unwrap();

    let session = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
        .with_intra_threads(1).unwrap()
        .commit_from_file(
            Path::new(PROJ_DIR).join("data").join("embedding_model.onnx")
        ).unwrap();

    let tokenizer = Tokenizer::from_file(
        Path::new(PROJ_DIR)
            .join("data")
            .join("tokenizer.json")
    ).unwrap();

    Inference {
        session,
        tokenizer
    }
}

fn mean_pooling(token_embeddings: &Array2<f32>) -> Array1<f32> {
    let mean_tokens = token_embeddings.mean_axis(Axis(0)).unwrap();
    // Normalizing the array to make it suitable for cosine similarity comparisons
    // by creating an array with a magnitude of one.
    let l2_norm = mean_tokens.iter().map(|x| x * x).sum::<f32>().sqrt();
    mean_tokens.iter().map(|x| x / l2_norm).collect()
}

fn embed(text: &str, inference: &Inference) -> Array1<f32> {
    let tokens = inference.tokenizer.encode(text, false).unwrap();
    let tokens = Array1::from_iter(tokens
        .get_ids()
        .iter()
        .map(|i| *i as i64)
    );

    let input_ids = tokens
        .view()
        .insert_axis(Axis(0));
    let token_type: Array<i64, Ix2> = Array::zeros((1, input_ids.shape()[1]));
    let attention_mask: Array<i64, Ix2> = Array::ones((1, input_ids.shape()[1]));

    let outputs = inference.session.run(
        inputs![
            "input_ids" => input_ids,
            "token_type_ids" => token_type,
            "attention_mask" => attention_mask
        ].unwrap()
    ).unwrap();

    let generated_tokens:  Array2<f32> = outputs["output"].try_extract_tensor().unwrap()
        .remove_axis(Axis(0))
        .to_owned()
        .into_dimensionality::<Ix2>()
        .unwrap();

    mean_pooling(&generated_tokens)
}


fn main() {
    let inference: Inference = setup_inference();
    let document = "This is Deven96 and HAKSOAT";
    let embeddings = embed(&document, &inference);

    println!("Text: {}", document);
    println!("Embedding shape: {:?}", embeddings.shape());
    println!("Embedding: {}", format!("{}", embeddings));
}
