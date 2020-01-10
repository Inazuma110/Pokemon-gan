# Pokemon-gan
Generate Pokemon using some GANs.

# Result
## DCGAN
![dcgan1](./result/dcgan/0000015003.png)
![dcgan2](./result/dcgan/0000014970.png)

## VAE
![vae](./result/vae/vae_sample.png)

## CGAN
![cgan](./result/cgan/15100.png)


# Usage
1. `cd Pokemon-gan`
1. Collect 64x64x3 Pokemon image into `poke64` directory.

## DCGAN
1. `pipenv run python3 ./dcgan.py`

## VAE
1. `pipenv run python3 ./vae/vae.py`

## CGAN
1. `cd cgan/`
1. Create `./cgan/info.pickle`. This file has image path and Pokemon's types as `pd.DataFrame`.
1. `pipenv run python3 cgan.py`
