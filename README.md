# Auxiliary Classifier GAN (AC-GAN)

This is a tensorflow implementation of the Auxiliary Classifier GAN described in the article [CONDITIONAL IMAGE SYNTHESIS WITH AUXILIARY CLASSIFIER GANS](https://arxiv.org/abs/1610.09585).

## Run

```
python train.py \
--dataset=mnist \
--input_height=64 \
--input_width=64 \
--input_channels=1 \
--output_height=64 \
--output_width=64 \
--log_dir=mnist_log_dir
```

## Examples
![samples]
(samples/sampled_images_680.jpg)

![samples]
(samples/sampled_images_690.jpg)