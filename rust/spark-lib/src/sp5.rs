pub enum Activation {
    ReLU,
    LeakyReLU,
    None,
}

pub fn contract_to_unisphere(x: f32, y: f32, z: f32) -> [f32; 3] {
    let mag = (x * x + y * y + z * z).sqrt();
    let (mut cx, mut cy, mut cz) = (x, y, z);
    if mag > 1.0 {
        if mag.is_infinite() {
            cx = 0.0;
            cy = 0.0;
            cz = 0.0;
        } else {
            let scale = (2.0 - 1.0 / mag) / mag;
            cx *= scale;
            cy *= scale;
            cz *= scale;
        }
    }
    [cx / 4.0 + 0.5, cy / 4.0 + 0.5, cz / 4.0 + 0.5]
}

pub fn get_tcnn_frequency_encoding(coords: [f32; 3], num_freqs: usize, out: &mut [f32]) {
    let mut idx = 0;
    for &val in &coords {
        for freq in 0..num_freqs {
            let input_val = val * 2.0f32.powi(freq as i32) * std::f32::consts::PI;
            out[idx] = input_val.sin();
            out[idx + 1] = input_val.cos();
            idx += 2;
        }
    }
    while idx < 96 {
        out[idx] = 1.0;
        idx += 1;
    }
}

pub fn run_tcnn_mlp(
    input: &[f32],
    flat_params: &[f32],
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
    num_hidden_layers: usize,
    activation: Activation,
) -> Vec<f32> {
    let pad16 = |val: usize| -> usize { ((val + 15) / 16) * 16 };
    let padded_in = pad16(in_dim);
    let padded_hidden = pad16(hidden_dim);
    let padded_out = pad16(out_dim);

    let max_len = padded_hidden.max(padded_in).max(padded_out);
    let mut current = vec![0.0f32; max_len];
    let mut next = vec![0.0f32; max_len];

    for i in 0..in_dim.min(max_len) {
        current[i] = input[i];
    }

    let apply_activation = |val: f32| -> f32 {
        match activation {
            Activation::ReLU => val.max(0.0),
            Activation::LeakyReLU => {
                if val > 0.0 {
                    val
                } else {
                    val * 0.01
                }
            }
            Activation::None => val,
        }
    };

    let mut offset = 0;

    // First layer: input to hidden
    for out_idx in 0..padded_hidden {
        let mut sum = 0.0;
        for in_idx in 0..padded_in {
            sum += current[in_idx] * flat_params[offset + out_idx * padded_in + in_idx];
        }
        next[out_idx] = apply_activation(sum);
    }
    offset += padded_in * padded_hidden;

    // Hidden layers
    for _ in 1..num_hidden_layers {
        for i in 0..padded_hidden {
            current[i] = next[i];
        }
        for out_idx in 0..padded_hidden {
            let mut sum = 0.0;
            for in_idx in 0..padded_hidden {
                sum += current[in_idx] * flat_params[offset + out_idx * padded_hidden + in_idx];
            }
            next[out_idx] = apply_activation(sum);
        }
        offset += padded_hidden * padded_hidden;
    }

    // Output layer
    for i in 0..padded_hidden {
        current[i] = next[i];
    }
    for out_idx in 0..padded_out {
        let mut sum = 0.0;
        for in_idx in 0..padded_hidden {
            sum += current[in_idx] * flat_params[offset + out_idx * padded_hidden + in_idx];
        }
        next[out_idx] = sum;
    }

    let mut out = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        out[i] = next[i];
    }
    out
}

pub fn run_pytorch_mlp(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    apply_relu: bool,
) -> Vec<f32> {
    let out_dim = biases.len();
    let in_dim = input.len();
    let mut out = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = biases[i];
        for j in 0..in_dim {
            sum += input[j] * weights[i * in_dim + j];
        }
        out[i] = if apply_relu { sum.max(0.0) } else { sum };
    }
    out
}

pub fn stable_lexicographic_sort(xyz: &[[f32; 3]]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..xyz.len()).collect();
    indices.sort_by(|&a, &b| {
        let pt_a = xyz[a];
        let pt_b = xyz[b];
        
        let res = pt_a[2].partial_cmp(&pt_b[2]).unwrap_or(std::cmp::Ordering::Equal);
        if res != std::cmp::Ordering::Equal {
            return res;
        }
        let res = pt_a[1].partial_cmp(&pt_b[1]).unwrap_or(std::cmp::Ordering::Equal);
        if res != std::cmp::Ordering::Equal {
            return res;
        }
        pt_a[0].partial_cmp(&pt_b[0]).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}
