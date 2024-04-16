
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*
};

#[derive(Clone)]
pub struct MnistBatcher<B: Backend>{
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images: Vec<Tensor<B, 3>> = items // Takes items from the Vec<MnistItem>
            .iter() // Creates an iterator over the items
            .map(| item | Data::<f32, 2>::from(item.image)) // For each item, convert the image to float32 data struct
            .map(| data | Tensor::<B, 2>::from_data(data.convert(), &self.device)) // for each data struct, create a tensor on the device
            .map(| tensor | tensor.reshape([1, 28, 28])) // For each tensor, reshape to the dimensions [C, H, W]
            // Normalize: Make each pixel between [0, 1] and make the mean=0, std=1
            // Values mean=0.1307, std=0.3081 are from the official PyTorch example
            .map(| tensor | ((tensor / 255) - 0.1307) / 0.3081) // For each Tensor, Apply Normalization
            .collect(); // Consume the resulting iterator & Collect the values into a new vector

        let targets = items
            .iter()
            .map(| item| Tensor::<B, 1, Int>::from_data(
                Data::from([(item.label as i64).elem()]),
                &self.device
            ))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MnistBatch { images, targets }
    }
}