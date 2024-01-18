#import pynput
import pyautogui




"""
for _ in range(10):
     pyautogui.move(-20, 10, duration=0.4)
     time.sleep(0.5)
     """


def move_mouse(x, y, speed = 0.5, iden_range = 5):

    if abs(x) > iden_range or abs(y) > iden_range:
        x, y = int(x*-3), int(y*3)
        try:
            pyautogui.move(y, x, duration=speed)
        except:
            pass

  
    

