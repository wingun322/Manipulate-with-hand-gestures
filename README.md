## Manipulate with hand gestures
![demonstration_video](/image/demonstration_video.gif)   

## Requirement
- mediapipe
- pyautogui
  
### main.py
This is a project program.<br>
This can be done to manipulate the computer using hand gestures.
### tools.py
It contains the functions required to run main.py .
### model/keypoint_classifier.py
It stores files related to finger gesture recognition.
### utils/cvfpscalc.py
This is a module for FPS measurement.
## Requirement
1. Move<br>
![MOVE](/image/MOVE.JPG)<br>
You can move your mouse when you blow your hand.<br>
2. Click<br>
![CLICK](/image/CLICK.JPG)<br>
You can click only if your index finger is open.<br>
3. Scroll<br>
![SCROLL](/image/SCROLL.JPG)<br>
If the index and middle fingers are open and the finger is moved up while moving the finger up, the scroll moves up, and if it moves down, the scroll moves down.<br>
4. Volume Down<br>
![V_DOWN](/image/V_DOWN.JPG)<br>
If you point your index finger to the left, the volume decreases.<br>
5. Volume Up<br>
![V_UP](/image/V_UP.JPG)<br>
If you open your thumb and point to the right, the volume will increase.<br>
6. Task Switching(alt + tab)<br>
![TAB](/image/TAB.JPG)<br>
If you show the back of your hand, it performs the task switching function.<br>
7. End Window(alt + f4)<br>
![TERMINATE](/image/TERMINATE.JPG)<br>
Just open your little finger and close the window.<br>
8. Go to desktop<br>
![HOME](/image/HOME.JPG)<br>
If you point your index finger down with only the index finger pinned, you will move to the desktop..<br>
## Model
The model for detecting the shape of the hand used the model in the link below.
- [MODEL](https://github.com/fredericomcorda/hand-gesture-recognition-mediapipe)
## REFERENCE
* [MediaPipe](https://mediapipe.dev/)
* [fredericomcorda/hand-gesture-recognition-mediapipe](https://github.com/fredericomcorda/hand-gesture-recognition-mediapipe)