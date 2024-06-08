# mlx-play

## Setup pyenv
The following commands only need to run once on the laptop.
```
# Make sure running in arm64 mode
arch

brew install pyenv
brew install pyenv-virtualenv
pyenv install 3.11.4
```

To clone and setup pyenv for this repo:
```
# Clone repo
git clone git@github.com:wenleix/mlx-play.git
cd mlx-play

# Setup pyenv
pyenv virtualenv 3.11.4 mlxplay-3.11.4
pyenv local mlxplay-3.11.4

# Install requirements
pip install -r requirements.txt
```
