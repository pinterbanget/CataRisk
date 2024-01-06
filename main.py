import os
import json
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
from tensorflow.keras.models import load_model


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(CURR_DIR, "assets", "Logo.png")
LONG_LOGO_PATH = os.path.join(CURR_DIR, "assets", "Long-Logo.png")
USER_LIST_PATH = os.path.join(CURR_DIR, "assets", "user_list.json")

IMAGE_THRESHOLD = 0.5
BLACK = "#121212"


class CataRiskApp():
    def __init__(self, root) -> None:
        '''
        Initializes the CataRisk application.

        Parameters
        ----------
        root : Tk
            The root Tkinter instance for the application.

        Attributes
        ----------
        root : Tk
            The root Tkinter instance for the application.
        current_user : str
            Stores the currently selected user.
        left_eye : ndarray or None
            Stores the left eye image.
        right_eye : ndarray or None
            Stores the right eye image.
        left_eye_result : str or None
            Stores the result of the left eye analysis.
        right_eye_result : str or None
            Stores the result of the right eye analysis.
        left_eye_image : PIL.Image or None
            Stores the left eye image in PIL format.
        right_eye_image : PIL.Image or None
            Stores the right eye image in PIL format.
        '''

        # Initialize Tkinter root
        self.root = root
        self.root.title("CataRisk Beta 1")
        self.root.configure(bg=BLACK, padx=30, pady=30)

        # Initialize variables to store eye images and analysis results
        self.current_user = "Unnamed"
        self.left_eye = None
        self.right_eye = None
        self.left_eye_result = None
        self.right_eye_result = None
        self.left_eye_image = None
        self.right_eye_image = None

        # Create and configure widgets for the application
        self.create_widgets()

    def create_widgets(self) -> None:
        '''
        Creates and configures various UI widgets for the application.

        Notes
        -----
        This method:
        - Sets the application icon and logo images.
        - Creates labels, dropdowns, buttons, and checkboxes for user interaction.
        - Initializes UI elements such as profile selection dropdown, profile creation button,
        webcam flip checkbox, and scan button.
        - Configures layout and positioning of these widgets within the application window.

        Returns
        -------
        None
        '''
        # Set icon and logo
        icon_image = Image.open(IMG_PATH)
        icon_image = icon_image.resize((200, 200))
        self.icon_photo = ImageTk.PhotoImage(icon_image)
        self.root.iconphoto(True, self.icon_photo)

        logo_image = Image.open(LONG_LOGO_PATH)
        logo_image = logo_image.resize((350, 140))
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        # Create a label for the logo
        self.logo_label = Label(self.root, image=self.logo_photo,
                                bg=BLACK, highlightthickness=0)
        self.logo_label.grid(column=0, row=0, columnspan=4, padx=10,
                            pady=10, sticky='n')

        user_list = self.read_user_list()

        # Create the profile selection dropdown
        self.user_value = StringVar(self.root)
        self.user_value.set('(Select profile...)')

        self.user_selection = ttk.Combobox(
            self.root, values=user_list, textvariable=self.user_value, state='readonly', width=25)
        self.user_selection.grid(column=0, row=1, sticky='w', padx=2, pady=2)

        # Create the Profile Creation button
        self.profile_creation_button = Button(self.root, text="Create Profile", command=self.create_profile_popup,
                                            highlightthickness=0, width=25)
        self.profile_creation_button.grid(column=2, row=1, sticky='e', padx=2, pady=2)

        # Create a style for the checkbox
        self.style = ttk.Style()
        self.style.configure("Sexy.TCheckbutton", background=BLACK, foreground="white")

        # Create the Flip Webcam checkbox
        self.webcam_flip = BooleanVar()
        self.flip_checkbox = ttk.Checkbutton(self.root, text="Flip Webcam", variable=self.webcam_flip, style="Sexy.TCheckbutton")
        self.flip_checkbox.grid(column=0, row=2, sticky='w', padx=2, pady=4)

        # Create the Scan button
        self.scan_button = Button(self.root, text="Start Scan", command=self.start_scan_process, highlightthickness=0, width=25)
        self.scan_button.grid(column=2, row=2, sticky='e', padx=2, pady=4)

    def read_user_list(self) -> list:
        '''
        Reads the user list from a file and returns it.

        Notes
        -----
        This method:
        - Attempts to read the user list from a specified file.
        - If the file doesn't exist, it creates an empty user list file.
        - Returns the retrieved user list or an empty list if no file is found.

        Returns
        -------
        list
            The user list retrieved from the file or an empty list if not found.
        '''

        # Continuously attempts to read the user list from a file
        while True:
            try:
                # Try to open and load the user list from the file
                with open(USER_LIST_PATH, "r") as f:
                    user_list = json.load(f)
                break

            except FileNotFoundError:
                # If the file doesn't exist, create an empty user list file
                with open(USER_LIST_PATH, "w") as f:
                    json.dump([], f)

        return user_list

    def create_profile_popup(self) -> None:
        '''
        Opens a popup window for creating and managing user profiles.

        Notes
        -----
        This method:
        - Creates a popup window to add, delete, and manage user profiles.
        - Provides functionality to add new users, delete existing users, and save changes.
        - Displays the list of existing users in a text box within the popup.
        - Saves the updated user list to a file when changes are made.
        - Updates the user selection dropdown with the modified user list.

        Returns
        -------
        None
        '''

        # Function to add a new user to the user list
        def add_name():
            new_user = user_entry.get().title()
            if new_user not in user_list:
                user_list.append(new_user)

                text_box.config(state=NORMAL)
                text_box.insert(END, new_user + "\n")  # Add the new user to the text box
                text_box.config(state=DISABLED)

                user_entry.delete(0, END)  # Clear the entry

            else:
                messagebox.showerror("Error", "User already exists.")
                popup.focus_set()

        # Function to delete an existing user from the user list
        def delete_name():
            delete_user = user_entry.get().title()
            if delete_user in user_list:
                user_list.remove(delete_user)

                text_box.config(state=NORMAL)
                text_box.delete(1.0, END)
                for user in user_list:
                    text_box.insert(END, user + "\n")
                text_box.config(state=DISABLED)
                
                user_entry.delete(0, END)

        # Function to save the updated user list to a file and update UI
        def save_profile():
            with open(USER_LIST_PATH, "w") as f:
                json.dump(user_list, f)
            self.user_selection["values"] = self.read_user_list()
            popup.destroy()

        # Create a popup window for managing user profiles
        popup = Toplevel(self.root)
        popup.wm_title("Add Profile")
        popup.configure(padx=15, pady=15, background=BLACK)

        # Labels, Entry, Buttons, and Text box for profile management
        select_label = Label(popup, text="Add Profile:", font=('Arial', 12, 'normal'),
                                foreground='White', background='black')
        select_label.grid(column=0, row=0, pady=2)

        user_entry = Entry(popup, width=20)
        user_entry.grid(column=1, row=0, padx=2, pady=2)
        user_entry.bind("<Return>", lambda event: add_name())

        save_button = Button(popup, text="Add", command=add_name)
        save_button.grid(column=2, row=0, padx=2, pady=2)

        delete_button = Button(popup, text="Delete", command=delete_name)
        delete_button.grid(column=3, row=0, padx=2, pady=2)

        text_box = Text(popup, height=10, width=40, wrap=WORD)
        text_box.grid(column=0, row=1, columnspan=4, padx=2, pady=10)

        user_list = self.read_user_list()
        for user in user_list:
            text_box.insert(END, user + "\n")

        text_box.config(state=DISABLED)

        select_class_button = Button(
            popup, text="Finish Profile Creation", command=save_profile)
        select_class_button.grid(column=0, row=2, columnspan=4, padx=2, pady=2)

    def detect_eyes(self, is_flipped: bool) -> None:
        '''
        Detects eyes from a live webcam feed using OpenCV (cv2) for face and eye detection.
        Stores the detected eye regions into the object.

        Parameters
        ----------
        is_flipped: bool
            Indicates if the default webcam feed needs to be horizontally flipped.

        Notes
        -----
        This function continuously captures frames from the webcam and detects eyes within detected faces.

        Steps:
        - Loads pre-trained cascade classifiers for face and eye detection.
        - Accesses the webcam feed (default or specified camera).
        - Processes each frame by converting it to grayscale and detecting faces.
        - Displays the live webcam feed with instructions for optimal eye scanning.
        - Identifies faces within the frame and, if two eyes are detected:
            - Extracts and sorts the eye regions within the face region.
            - Resizes the detected eye regions to a standard size (256x256).
            - Converts the eye regions from BGR to RGB format.
            - Stores the left and right eye regions in the object attributes.
        - Stops the process when the 'Esc' key is pressed or when two eyes are detected.

        Returns
        -------
        None
        '''
        # Load the pre-trained cascade classifiers for face and eye detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        # Access the webcam
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        
        while True:
            _, frame = cap.read()  # Read frame from the webcam

            # Flip the frame horizontally to counter webcam mirroring
            if is_flipped:
                frame = cv2.flip(frame, 1)

            # Convert the frame to grayscale for face and eye detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Display the live webcam feed
            cv2.putText(frame, "Remove glasses and use good lighting for better results", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("CataRisk - Eye Scanning", frame)
            cv2.waitKey(1)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]  # Region of Interest (ROI) in grayscale

                # Detect eyes within the face ROI
                eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=30)  # Adjust minNeighbors

                # Check if two eyes are detected
                if len(eyes) >= 2:
                    # Sort eyes by x-coordinate to determine left and right
                    eyes = sorted(eyes, key=lambda x: x[0])

                    # Extract the two eye regions
                    eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
                    eye2_x, eye2_y, eye2_w, eye2_h = eyes[1]

                    # Extract eye regions from the frame
                    eye1 = frame[y + eye1_y:y + eye1_y + eye1_h, x + eye1_x:x + eye1_x + eye1_w]
                    eye2 = frame[y + eye2_y:y + eye2_y + eye2_h, x + eye2_x:x + eye2_x + eye2_w]

                    # Resize the eye regions to 256x256
                    eye1_resized = cv2.resize(eye1, (256, 256))
                    eye2_resized = cv2.resize(eye2, (256, 256))

                    # Convert BGR to RGB
                    eye1_converted = cv2.cvtColor(eye1_resized, cv2.COLOR_BGR2RGB)
                    eye2_converted = cv2.cvtColor(eye2_resized, cv2.COLOR_BGR2RGB)
                
                    self.left_eye = eye1_converted
                    self.right_eye = eye2_converted

                    # Release the camera
                    cap.release()

                    return None
                
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break

        return None

    def detect_cataract(self) -> None:
        '''
        Uses a deep learning model to assess if the eyes exhibit cataract conditions.
        Stores the assessment results obtained from the model into the object.

        Notes
        -----
        This method loads a deep learning model trained for cataract detection.
        - Prepares the eye images for input to the model.
        - Utilizes the model to make predictions about cataract presence in the eyes.
        - Assigns the cataract assessment results to the respective eye attributes in text format ("OK" or "RISK").

        Returns
        -------
        None
        '''
        # Load the pre-trained deep learning model for cataract detection
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_model.h5')
        model = load_model(model_path)

        # Prepare eye images for input to the model
        input_data = np.stack([self.left_eye / 255.0, self.right_eye / 255.0], axis=0)

        # Utilize the model to make predictions
        predictions = model.predict(input_data)

        # Convert predictions to text format ("OK" or "RISK") based on a threshold
        prediction_to_text = ["OK" if pred > IMAGE_THRESHOLD else "RISK" for pred in predictions]

        # Assign the cataract assessment results to the respective eye attributes
        self.left_eye_result, self.right_eye_result = prediction_to_text
    
    def display_results(self) -> None:
        '''
        Creates a popup window displaying eye images and their prediction results.

        Notes
        -----
        This method prepares and displays left and right eye images along with their corresponding
        cataract prediction results in a Tkinter popup window.

        Returns
        -------
        None
        '''
        # Create a popup window
        popup = Toplevel()

        # Set the title of the popup window
        popup.title(f"CataRisk Results - {self.current_user}")

        # Convert numpy arrays to PIL Image objects
        self.left_eye_image = Image.fromarray(self.left_eye)
        self.right_eye_image = Image.fromarray(self.right_eye)

        # Convert PIL Image objects to PhotoImage format for displaying in Tkinter
        self.eye_images = {
            'left': ImageTk.PhotoImage(self.left_eye_image),
            'right': ImageTk.PhotoImage(self.right_eye_image)
        }

        # Create labels to display eye images and handle predictions simultaneously
        for eye, image in self.eye_images.items():
            # Create a label for displaying each eye image
            eye_label = Label(popup, image=image)
            eye_label.image = image  # Retain reference to the image object
            eye_label.grid(row=0, column=0 if eye == 'left' else 1, padx=5, pady=5)

            # Get the prediction result for the respective eye
            eye_prediction = self.left_eye_result if eye == 'left' else self.right_eye_result

            # Define the color based on the prediction result for displaying eye prediction labels
            color = "red" if eye_prediction == "RISK" else "green"

            # Create labels to display model predictions with formatted text based on each eye's result
            Label(popup, text=f"{eye.title()} Eye: {eye_prediction}", fg=color, font=("Arial", 12, "bold")).grid(
                row=1, column=0 if eye == 'left' else 1, padx=5, pady=5
            )

    def save_results(self) -> None:
        '''
        Saves the eye images in a designated folder with timestamped filenames.

        Notes
        -----
        This method creates a folder within the 'results' directory to save the eye images.
        - Generates a timestamp for filename uniqueness.
        - Constructs filenames containing eye type, cataract prediction, and timestamp.
        - Saves the left and right eye images with their respective prediction results in the designated folder.

        Returns
        -------
        None
        '''
        # Get the current timestamp for filename uniqueness
        current_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Create the directory path to save the results (if it doesn't exist)
        save_directory = os.path.join(CURR_DIR, "results", "_".join(self.current_user.split()))
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the left eye image with filename containing eye type, prediction, and timestamp
        self.left_eye_image.save(f"{save_directory}/{current_time}_LEFT_{self.left_eye_result}.png")

        # Save the right eye image with filename containing eye type, prediction, and timestamp
        self.right_eye_image.save(f"{save_directory}/{current_time}_RIGHT_{self.right_eye_result}.png")
        
    def start_scan_process(self) -> None:
        '''
        Initiates the entire eye scanning and assessment process.

        Notes
        -----
        This method:
        - Retrieves the selected user profile and webcam flip status.
        - Attempts to execute the eye scanning and assessment process.
        - Handles any TypeError exceptions that might occur during the process.

        Returns
        -------
        None
        '''

        # Retrieve the selected user profile and webcam flip status
        name = self.user_selection.get()
        if name != "(Select profile...)":
            self.current_user = name

        webcam_flipped = self.webcam_flip.get()

        try:
            # Execute the eye detection process using webcam feed
            self.detect_eyes(webcam_flipped)

            # Assess cataract presence in the detected eyes
            self.detect_cataract()

            # Display eye images and cataract prediction results in a popup window
            self.display_results()

            # Save eye images with cataract assessment results in a designated folder
            self.save_results()

        except TypeError:
            # Handle TypeError exceptions that might occur during the process
            pass

if __name__ == '__main__':
    root = Tk()
    app = CataRiskApp(root)
    root.geometry('435x320')
    root.minsize(435, 320)
    root.maxsize(435, 320)
    root.mainloop()