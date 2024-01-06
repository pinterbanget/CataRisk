# CataRisk

Cataract risk scanner from your webcam. Built using Python, with Tkinter as the GUI, OpenCV for image recognition, and TensorFlow as the framework for image classification.

## Dataset

Dataset is from <a href="https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset"> Kaggle</a>.

## Usage

1) Run the `.exe` file.
2) Select the `"Create Profile"` button to create a new profile. Write down your name and press `"Add"`. Alternatively, you can also press Enter on your keyboard.
3) Press the `"Finish Profile Selection"` button. Now, you can select your name from the dropdown menu.
4) Press the `"Start Scan"` button. Wait until a webcam window appears.
5) Stare directly at your webcam.
    * For best results:
        * Do not use glasses or any kind of eye covering.
        * Make sure your environment is well-lit.
        * Use a good-quality webcam. You can also try <a href="https://www.wired.com/story/use-your-phone-as-webcam/"> turning your phone into a webcam</a>.
6) The program will auto-detect your eyes. A result will be shown afterwards.
7) You can close the results window. The pictures with your eyes are saved in the `results` folder.

## Compiling

`nuitka --enable-plugin=tk-inter --onefile --windows-icon-from-ico=assets/Logo.ico --include-package-data=assets --disable-console main.py`

## Should I use this as a medical tool?

<b>Obviously not!</b> This project rather showcases a basic program you can create upon learning Python, Tkinter, OpenCV, and TensorFlow.
