import time

import evdev
from evdev import InputDevice
from select import select
import re

ctrlkeymap = {
    # define USB_HID_MODIFIER_LEFT_CTRL   0x01
    # define USB_HID_MODIFIER_LEFT_SHIFT  0x02
    # define USB_HID_MODIFIER_LEFT_ALT    0x04
    # define USB_HID_MODIFIER_LEFT_GUI    0x08 // (Win/Apple/Meta)
    # define USB_HID_MODIFIER_RIGHT_CTRL  0x10
    # define USB_HID_MODIFIER_RIGHT_SHIFT 0x20
    # define USB_HID_MODIFIER_RIGHT_ALT   0x40
    # define USB_HID_MODIFIER_RIGHT_GUI   0x80
    # define USB_HID_KEY_L     0x0F

    "KEY_LEFTSHIFT": 0x2,
    "KEY_RIGHTSHIFT": 0x20,
    "KEY_LEFTCTRL": 0x01,
    "KEY_RIGHTCTRL": 0x10,
    "KEY_LEFTALT": 0x4,
    "KEY_RIGHTALT": 0x40,
    "KEY_LEFTMETA": 0x8,
    "KEY_RIGHTMETA": 0x80,
}

normalkeymap = {
    "KEY_A": 0x04,
    "KEY_B": 0x05,
    "KEY_C": 0x06,
    "KEY_D": 0x07,
    "KEY_E": 0x08,
    "KEY_F": 0x09,
    "KEY_G": 0x0A,
    "KEY_H": 0x0B,
    "KEY_I": 0x0C,
    "KEY_J": 0x0D,
    "KEY_K": 0x0E,
    "KEY_L": 0x0F,
    "KEY_M": 0x10,
    "KEY_N": 0x11,
    "KEY_O": 0x12,
    "KEY_P": 0x13,
    "KEY_Q": 0x14,
    "KEY_R": 0x15,
    "KEY_S": 0x16,
    "KEY_T": 0x17,
    "KEY_U": 0x18,
    "KEY_V": 0x19,
    "KEY_W": 0x1A,
    "KEY_X": 0x1B,
    "KEY_Y": 0x1C,
    "KEY_Z": 0x1D,
    "KEY_0": 0x27,
    "KEY_1": 0x1E,
    "KEY_2": 0x1F,
    "KEY_3": 0x20,
    "KEY_4": 0x21,
    "KEY_5": 0x22,
    "KEY_6": 0x23,
    "KEY_7": 0x24,
    "KEY_8": 0x25,
    "KEY_9": 0x26,
    "KEY_KP0": 0x62,
    "KEY_KP1": 0x59,
    "KEY_KP2": 0x5A,
    "KEY_KP3": 0x5B,
    "KEY_KP4": 0x5C,
    "KEY_KP5": 0x5D,
    "KEY_KP6": 0x5E,
    "KEY_KP7": 0x5F,
    "KEY_KP8": 0x60,
    "KEY_KP9": 0x61,

    "KEY_KPASTERISK": 0x55,
    "KEY_KPPLUS": 0x56,
    "KEY_KPENTER": 0x58,
    "KEY_KPMINUS": 0x57,
    "KEY_KPDOT": 0x63,
    "KEY_KPSLASH": 0x54,
    "KEY_F1": 0x3A,
    "KEY_F2": 0x3B,
    "KEY_F3": 0x3C,
    "KEY_F4": 0x3D,
    "KEY_F5": 0x3E,
    "KEY_F6": 0x3F,
    "KEY_F7": 0x40,
    "KEY_F8": 0x41,
    "KEY_F9": 0x42,
    "KEY_F10": 0x43,
    "KEY_F11": 0x44,
    "KEY_F12": 0x45,

    "KEY_BACKSPACE": 0x2A,
    "KEY_TAB": 0x2B,
    "KEY_ENTER": 0x28,
    "KEY_CAPSLOCK": 0x39,
    "KEY_ESC": 0x29,
    "KEY_SPACE": 0x2C,
    "KEY_PAGEUP": 0x4B,
    "KEY_PAGEDOWN": 0x4E,
    "KEY_END": 0x4D,
    "KEY_HOME": 0x4A,
    "KEY_LEFT": 0x50,
    "KEY_UP": 0x52,
    "KEY_RIGHT": 0x4F,
    "KEY_DOWN": 0x51,
    "KEY_INSERT": 0x49,
    "KEY_DELETE": 0x4C,
    "KEY_NUMLOCK": 0x53,
    "KEY_GRAVE": 0x35,
    "KEY_BACKSLASH": 0x31,
    "KEY_LEFTBRACE": 0x2F,
    "KEY_RIGHTBRACE": 0x30,
    "KEY_SLASH": 0x38,
    "KEY_COMMA": 0x36,
    "KEY_DOT": 0x37,
    "KEY_SEMICOLON": 0x33,
    "KEY_APOSTROPHE": 0x34,
    "KEY_MINUS": 0x2D,
    "KEY_EQUAL": 0x2E,
    "KEY_SYSRQ": 0x46,
    "KEY_SCROLLLOCK": 0x47,
    "KEY_PAUSE": 0x48,
}


def ispress(str):
    if str:
        if str.find('down') != -1:
            return True
        if str.find('hold') != -1:
            return True
        if str.find('up') != -1:
            return False


def detectInputKey(devpath, msgq):
    dev = InputDevice(devpath)
    keylist = [0] * 9
    print(keylist)
    while True:
        for event in dev.read_loop():

            if event.type == evdev.ecodes.EV_KEY:
                msg = str(evdev.categorize(event))
                sendstr = re.findall(r"(\([^\)]*\))", msg)

                if len(sendstr):
                    matchkey = sendstr[0][1:-1]
                    print("matchkey:", matchkey)
                    ctrkeyval = ctrlkeymap.get(matchkey)
                    print("ctr key val:", ctrkeyval)
                    keval = normalkeymap.get(matchkey)
                    print("key val:", keval)

                if ctrkeyval:  # ctrl key
                    if ispress(msg):
                        keylist[0] = keylist[0] | ctrkeyval
                    else:
                        keylist[0] = keylist[0] & (~ctrkeyval)

                elif keval:  # normal key
                    if ispress(msg):
                        for i in range(2, 8):
                            if keylist[i] == 0 or keylist[i] == keval:
                                keylist[i] = keval
                                break
                    else:
                        for i in range(2, 8):
                            if keylist[i] == keval:
                                keylist[i] = 0
                                break

                keylist[8] = 0xff
                byt = bytes(keylist)
                print(keylist, byt)

                if msgq:
                    msgq.put(byt)


def IsAnyKeyPress():

    dev = InputDevice('/dev/input/event4').read_one()

    if dev:
        for event in dev:
            if event.type == evdev.ecodes.EV_KEY:
                msg = str(evdev.categorize(event))
                sendstr = re.findall(r"(\([^\)]*\))", msg)

                if len(sendstr):
                    matchkey = sendstr[0][1:-1]
                    if normalkeymap.get(matchkey) or ctrlkeymap.get(matchkey):
                        return True
    return False


if __name__ == "__main__":
    detectInputKey('/dev/input/event4', 0)