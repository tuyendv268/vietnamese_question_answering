import os
import time
from dataclasses import dataclass
import pickle as pkl
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException


class Color:
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    YELLOW = "\033[1;33m"
    WHITE = "\033[1;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_GRAY = "\033[0;37m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"


class Style:
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"


class WebsiteUpdatedError(Exception):
    pass


def web_updated_error():
    return WebsiteUpdatedError("Poe may be updated! Please contact to hlongng2002241 for updated crawler")


@dataclass
class XPaths:
    WELCOME_BUTTON = "/html/body/div/div[1]/div[2]/a"
    EMAIL_TEXT_INPUT = "/html/body/div[1]/main/div/div[2]/form/input"
    OTP_TEXT_INPUT = "/html/body/div[1]/main/div/div[3]/form/input"
    OTP_RESENT_BUTTON = "/html/body/div[1]/main/div/button[1]"
    OTP_ERROR_LABEL = "/html/body/div[1]/main/div/div[3]/div"
    PROMPT_TEXT_AREA = "/html/body/div[1]/div[1]/div/section/div[2]/div/div/footer/div/div/div/textarea"
    PROMPT_SENT_BUTTON = "/html/body/div[1]/div[1]/div/section/div[2]/div/div/footer/div/div/button[2]"
    CHAT_PAIRS = "/html/body/div[1]/div[1]/div/section/div[2]/div/div/div[1]/div"
    RESPONSE_TEXT = "div[2]/div[2]/div[1]/div[1]/div"
    CLEAR_PROMPT_BUTTON = "/html/body/div[1]/div[1]/div/section/div[2]/div/div/footer/div/button"
    SAGE_BUTTON = "/html/body/div[1]/div[1]/div/aside/div/menu/section[1]/li[1]/a"
    CLAUDE_INSTANT_BUTTON = "/html/body/div[1]/div[1]/div/aside/div/menu/section[1]/li[5]/a"
    CHATGPT_BUTTON = "/html/body/div[1]/div[1]/div/aside/div/menu/section[1]/li[6]/a"


@dataclass
class Texts:
    OTP_ERROR_LABEL = "The code you entered is not valid. Please try again."


class PoeChatBot:

    def __init__(self, reset_cookies: bool=True) -> None:
        self.driver = webdriver.Chrome()
        self.driver.get("https://poe.com/Sage")
        self.xpaths = XPaths()
        self.texts = Texts()

        self.models = {
            "sage": self.xpaths.SAGE_BUTTON,
            "claude": self.xpaths.CLAUDE_INSTANT_BUTTON,
            "chatgpt": self.xpaths.CHATGPT_BUTTON
        }

        self.cookie_path = "data/cookies/cookies.pkl"
        if os.path.exists(self.cookie_path) is False or reset_cookies:
            os.makedirs(os.path.dirname(os.path.abspath(self.cookie_path)), exist_ok=True)
            self.sign_in()
        else:
            self.load_cookies()
            self.wait_and_get(self.xpaths.WELCOME_BUTTON).click()

    def sign_in(self):
        # self.wait_and_get(self.xpaths.WELCOME_BUTTON).click()
        self.wait_and_get(self.xpaths.WELCOME_BUTTON).click()

        # enter email:
        self.log("Enter your email:", end=" ", color=Color.GREEN)
        email = input()
        self.wait_and_get(self.xpaths.EMAIL_TEXT_INPUT).send_keys(email, Keys.ENTER)
    
        # enter OTP token
        otp_input = self.wait_and_get(self.xpaths.OTP_TEXT_INPUT)
        while True:
            self.log(f"Enter your OTP code that poe.com sent to {email}:", end=" ", color=Color.GREEN)
            otp = input()
            otp_input.send_keys(otp, Keys.ENTER)
            time.sleep(0.5)

            if self.check_exists_by_xpath(self.xpaths.OTP_ERROR_LABEL):
                text = self.wait_and_get(self.xpaths.OTP_ERROR_LABEL).text
                if text != self.texts.OTP_ERROR_LABEL:
                    raise web_updated_error()
                
                self.log("Your OTP code is incorrect", color=Color.RED)
                otp_input.clear()
                
                if self.yes_no("Do you want to resend OTP code?", color=Color.GREEN):
                    self.wait_and_get(self.xpaths.OTP_RESENT_BUTTON).click()
                    self.log("Please check your email again ...")
            else:
                break

        pkl.dump(self.driver.get_cookies(), open(self.cookie_path, "wb"))

    def load_cookies(self):
        cookies = pkl.load(open(self.cookie_path, "rb"))
        for c in cookies:
            self.driver.add_cookie(c)

    def check_exists_by_xpath(self, xpath: str):
        try:
            self.driver.find_element(By.XPATH, xpath)
        except NoSuchElementException:
            return False
        return True

    def wait_and_get(self, xpath, parent=None, multiple: bool=False, time_out: float=10):
        if parent is None:
            parent = self.driver
        try:
            WebDriverWait(parent, time_out).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            if not multiple:
                return parent.find_element(By.XPATH, xpath)
            else:
                return parent.find_elements(By.XPATH, xpath)
        except NoSuchElementException:
            raise web_updated_error()
        
    def log(self, *args, color=None, end="\n", sep=" "):
        if color is None:
            color = Color.WHITE
        print(color, end="")
        print(*args, end=end, sep=sep)
        print(Color.WHITE, end="")

    def yes_no(self, question: str, color=None):
        while True:
            self.log(question, "[y/n]", end=" ", color=color)
            a = input().lower()
            if a == "y":
                return True
            if a == "n":
                return False

    def push_prompt(self, prompt: str, beautify: bool=False, clear: bool=False):
        if clear:
            self.clear_old_prompts()

        self.wait_and_get(self.xpaths.PROMPT_TEXT_AREA).send_keys(prompt, Keys.ENTER)
        time.sleep(2)

        try:
            WebDriverWait(self.driver, 60).until(
                EC.element_to_be_clickable((By.XPATH, self.xpaths.PROMPT_SENT_BUTTON))
            )
            chat_pairs = self.wait_and_get(self.xpaths.CHAT_PAIRS, multiple=True)
            latest = chat_pairs[-1]
            response = self.wait_and_get(self.xpaths.RESPONSE_TEXT, latest)
            if not beautify:
                return response.text
            else:
                return self.beautify(response)
        except NoSuchElementException:
            return None
    
    def beautify(self, element):
        html = element.get_attribute("outerHTML")
        try:
            import html2text
        except:
            raise RuntimeError("Not found package 'html2text', please install it: pip install html2text")
        h = html2text.HTML2Text()
        return h.handle(html).strip()
            
    def clear_old_prompts(self):
        self.wait_and_get(self.xpaths.CLEAR_PROMPT_BUTTON).click()

    def change_bot(self, name: str):
        if name not in list(self.models.keys()):
            return False
        self.wait_and_get(self.models[name]).click()
        return True

    def quit(self):
        self.driver.quit()

    def chat(self):
        docs = f"""Welcome to poe chat
        
Here are some basic commands:
    :q = quit
    :c = clear old prompt
    :b = toggle beauty mode
    :m MODEL_NAME = change model, currently we have {list(self.models.keys())}
        """
        self.log(docs, color=Color.YELLOW)
        bot_name = "sage"
        beautify = True
        while True:
            self.log("User: ", end=" ", color=Color.GREEN)
            prompt = input()

            if prompt.strip().lower() == ":q":
                return
            elif prompt.strip().lower() == ":c":
                self.clear_old_prompts()
                self.log("System: Old prompts cleared", color=Color.YELLOW)
            elif prompt.strip().lower() == ":b":
                beautify = not beautify
                self.log("System: beatify =", beautify, color=Color.YELLOW)
            elif prompt.strip().lower().startswith(":m"):
                bot_name = prompt.lower().split()[1]
                if self.change_bot(bot_name):
                    self.log(f"System: change to model '{bot_name}'", color=Color.YELLOW)
                else:
                    self.log(f"System: model '{bot_name}' not found", color=Color.YELLOW)
            else:
                response = self.push_prompt(prompt, beautify=beautify)
                self.log(f"Bot {bot_name}:", color=Color.BLUE, end=" ")
                self.log(response + "\n")

if __name__ == "__main__":
    poe = PoeChatBot()
    poe.chat()
    poe.quit()