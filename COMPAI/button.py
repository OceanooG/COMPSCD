import pygame
class Button():
    def __init__(self, screen, msg, left,top):  # msg为要在按钮中显示的文本
        self.screen = screen
        self.screen_rect = screen.get_rect()

        self.width, self.height = 150, 50
        self.button_color = (72, 61, 139)
        self.text_color = (255, 255, 255)
        pygame.font.init()
        self.font = pygame.font.SysFont('kaiti', 20)

        self.rect = pygame.Rect(0, 0, self.width, self.height)
        #self.rect.center = self.screen_rect.center
        self.left = left
        self.top = top

        self.deal_msg(msg)  # 渲染图像

    def deal_msg(self, msg):

        self.msg_img = self.font.render(msg, True, self.text_color, self.button_color)
        self.msg_img_rect = self.msg_img.get_rect()
        self.msg_img_rect.center = self.rect.center

    def draw_button(self):
        #self.screen.fill(self.button_color, self.rect)
        self.screen.blit(self.msg_img, (self.left,self.top))

    def is_click(self):
        point_x, point_y = pygame.mouse.get_pos()
        x = self.left
        y = self.top
        w, h = self.msg_img.get_size()

        in_x = x < point_x < x + w
        in_y = y < point_y < y + h
        return in_x and in_y

