# MLAnalogClock - Keras model setup, training, and mlmodel generation for reading analog clocks

## Setup (MacOS)

1. Install nodejs & npm <br />
`brew install node`

2. Install ciaro <br />
`brew install cairo`


3. Intall libjpeg <br />
`brew install libjpeg`


4. Intall pango <br />
`brew install pango`


5. Intall giflib <br />
`brew install giflib`

## Usage

### Generate training and validation data

* From within /MLAnalogCloc folder, run <br />
`node generateImages.js`

### Model creation and training

```
sudo python3 train.py
```

### Testing

```
python3 predict.py
```
