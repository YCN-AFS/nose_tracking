#import pynput
import pyautogui





def move_mouse(x, y, speed = 0.1, iden_range = 4):
    average = (abs(x)+abs(y)) /2

    if average >= iden_range:
        if iden_range <= average <= 6:  
            x, y = int(x*-1), int(y)
            try:
                pyautogui.move(y, x, duration=speed)
            except:
            #có bug chưa fix ở đây
                pass
  
        else:
            x, y = int(x*-2.5), int(y*2.5)

            try:
                pyautogui.move(y, x, duration=speed)
            except:
                #có bug chưa fix ở đây
                pass
         

def scroll_mose(x, strength=2):

    if x > -22:
        pyautogui.scroll(-strength)
        print("Down")
    elif x < 22:
        print("Up")
        pyautogui.scroll(strength)

        


    


















  
    

