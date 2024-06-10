import random
import numpy as np
import pygame 


class Application:
    def __init__(self):
        self.size = self.width, self.height = 640, 480
    
    def on_init(self):
        pygame.init()
        pygame.display.set_caption("Random Walk")
        self.display = pygame.display.set_mode(self.size)
        self.isRunning = True
    
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.isRunning = False
    
    def on_loop(self):
        pass
    
    def on_render(self):
        pass
    
    def on_execute(self):
        if self.on_init() == False:
            self.isRunning = False
            
        while self.isRunning:
            for event in pygame.event.get():
                self.on_event(event)
            
            self.on_loop()
            self.on_render()
        
        pygame.quit()
if __name__ == "__main__":
    t = Application()
    t.on_execute()