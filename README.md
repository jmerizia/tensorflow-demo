# Tensorflow Demo

Slides: [put link to slides here]

## Setup

### Windows:

1. Install python3: https://www.python.org/ftp/python/3.5.2/python-3.5.2-amd64.exe
**NOTE: Make sure to check the box that says ADD PYTHON TO SYSTEM PATH**
2. Open command prompt
3. Type `pip3 install --upgrade tensorflow`
4. Clone this repository: `git clone https://github.com/jmerizia/tensorflow-demo`
5. `cd tensorflow-demo`
6. Install necessary libraries:
- `pip3 install --upgrade tensorflow`
- `pip3 install --upgrade flask`
- `pip3 install --upgrade matplotlib`
- `pip3 install --upgrade scikit-image`
- `pip3 install --upgrade pandas`
7. Download Sublime Text 3: https://download.sublimetext.com/Sublime%20Text%20Build%203143%20x64%20Setup.exe
**NOTE: You can use another editor (just not Notepad)**
8. Open `hello.py` with Sublime Text 3 (or another editor)
9. Type in `python hello.py` to run the Hello World program

### MacOS or Linux: (using virtualenv)
1. If you don't have python3, go to: https://www.python.org/downloads/ and download
**python 3.6**
1. `sudo easy_install pip`
2. `sudo easy_install virtualenv`
3. Clone this repository: `git clone https://github.com/jmerizia/tensorflow-demo`
4. `cd tensorflow-demo`
5. `virtualenv --system-site-packages .` **(DON'T FORGET THE ".")**
6. `source ./bin/activate`
**NOTE: Now your prompt should look something like this:** `(tensorflow-demo)$`
7. Install necessary libraries: **If it doesn't work, try with** `sudo`
- `pip3 install --upgrade tensorflow`
- `pip3 install --upgrade flask`
- `pip3 install --upgrade matplotlib`
- `pip3 install --upgrade scikit-image`
- `pip3 install --upgrade pandas`
8. Download Sublime Text 3: http://www.sublimetext.com/3
**NOTE: You can use another editor (just not Notepad)**
9. Open `hello.py` with Sublime Text 3 (or another editor)
10. Type in `python3 hello.py` to run the Hello World program
**After the workshop, type in** `deactivate` **to deactivate virtualenv**

## Links/Resources:

- More on installing pip on linux:
https://www.tecmint.com/install-pip-in-linux/
- TensorFlow Website:
https://www.tensorflow.org/
- MNIST Database of Handwritten Digits:
http://yann.lecun.com/exdb/mnist/
- Tons of Tensorflow Examples:
https://github.com/aymericdamien/TensorFlow-Examples
