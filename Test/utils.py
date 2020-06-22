import matplotlib.pyplot as plt
import torch
import numpy as np
import pygame

def draw_plot(x, y, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, show=False):

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    if xlabel is not None:
        # naming the x axis
        plt.xlabel(xlabel)
    if ylabel is not None:
        # naming the y axis
        plt.ylabel(ylabel)
    if title is not None:
        # giving a title to my graph
        plt.title(title)

    # plotting the points
    plt.plot(x, y)

    if show:
        # # function to show the plot
        plt.show()

def draw_grid(grid_size, window_size, state_action_values=None, all_actions=None):
    ground_color = [255, 255, 255]
    # agent_color = [i * 255 for i in self._agent_color]
    # ground_color = [i * 255 for i in self._ground_color]
    # obstacle_color = [i * 255 for i in self._obstacle_color]
    text_color = (240,240,10)
    info_color = (200, 50, 50)
    # This sets the WIDTH and HEIGHT of each grid location
    WIDTH = int(window_size[0] / grid_size[1])
    HEIGHT = int(window_size[1] / grid_size[0])

    # This sets the margin between each cell
    MARGIN = 1


    # Initialize pygame
    pygame.init()

    # Set the HEIGHT and WIDTH of the screen
    WINDOW_SIZE = [window_size[0], window_size[1]]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # Set title of screen
    pygame.display.set_caption("Grid")

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    font = pygame.font.Font('freesansbold.ttf', 20)
    info_font = pygame.font.Font('freesansbold.ttf', 20)


    done = False
    # -------- Main Program Loop -----------
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True

        # Set the screen background
        screen.fill((100,100,100))


        # Draw the grid
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                color = ground_color
                # if list(grid[x][y]) == self._agent_color:
                #     color = agent_color
                # elif list(grid[x][y]) == self._obstacle_color:
                #     color = obstacle_color
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * y + MARGIN,
                                  (MARGIN + HEIGHT) * x + MARGIN,
                                  WIDTH,
                                  HEIGHT])
                if state_action_values is not None:
                    # showing values only for 4 basic actions
                    up_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                      (MARGIN + HEIGHT) * x + MARGIN]
                    up_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                      (MARGIN + HEIGHT) * x + MARGIN]
                    down_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                       (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                    down_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                        (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                    center = [(up_right_corner[0] + up_left_corner[0]) // 2,
                              (up_right_corner[1] + down_right_corner[1]) // 2]

                    pygame.draw.polygon(screen, info_color,
                                        [up_left_corner, up_right_corner, center],
                                        1)
                    pygame.draw.polygon(screen, info_color,
                                        [up_right_corner, down_right_corner, center],
                                        1)
                    pygame.draw.polygon(screen, info_color,
                                        [down_right_corner, down_left_corner, center],
                                        1)
                    pygame.draw.polygon(screen, info_color,
                                        [down_left_corner, up_left_corner, center],
                                        1)
                    for a in all_actions:
                        if tuple(a) == (0,1):
                            right = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (1,0):
                            down = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (0,-1):
                            left = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        elif tuple(a) == (-1,0):
                            up = info_font.render(str(state_action_values[(x,y), tuple(a)]), True, info_color)
                        else:
                            raise ValueError("action cannot be rendered")
                    screen.blit(left,
                               (up_left_corner[0] , # + 0.5 * left.get_rect().width
                                center[1])) #left
                    screen.blit(right,
                               (up_right_corner[0] - right.get_rect().width,
                                center[1]))  # right
                    screen.blit(up,
                               (center[0] - up.get_rect().width // 2,
                                center[1] - 10 - up.get_rect().width))  # up
                    screen.blit(down,
                               (center[0] - down.get_rect().width // 2,
                                center[1] + 20 + down.get_rect().width - down.get_rect().height))  # down

        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
