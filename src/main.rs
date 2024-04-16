use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use burn::data::dataloader::Dataset;
use my_first_rust_DL_app::model::ModelConfig;


fn main() {
    type ModelBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type ModelAutodiffBackend = Autodiff<ModelBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    my_first_rust_DL_app::training::train::<ModelAutodiffBackend>(
        "/tmp/my_first_rust_DL_app",
        my_first_rust_DL_app::training::TrainingConfig::new(
            ModelConfig::new(10, 512),
            AdamConfig::new(),
        ),
        device,
    );

    // my_first_rust_DL_app::inference::infer::<ModelBackend>(
    //     "/tmp/my_first_rust_DL_app",
    //     device,
    //     burn::data::dataset::vision::MnistDataset::test()
    //         .get(42)
    //         .unwrap(),
    // );

}