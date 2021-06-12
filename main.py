import pygame
import random
import neat
from math import hypot
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
FPS = 30

PLAYER_IMG = pygame.image.load("assets/drone.png")
PLAYER_IMG_HEIGHT = PLAYER_IMG.get_height()
PLAYER_IMG_WIDTH = PLAYER_IMG.get_width()

GRAVITY = 0
DRAG = 0.5

# Initialize pygame
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)


# Player
class Drone:

    def __init__(self, pos=(340, 200)):
        self.img = PLAYER_IMG
        # coordinates
        self.x = pos[0]
        self.y = pos[1]
        # velocity
        self.y_vel = 0
        self.x_vel = 0
        # velocity boundaries
        self.y_vel_up_boundary = -5
        self.y_vel_down_boundary = 5
        # acceleration
        self.y_acc = 0
        self.x_acc = 0
        # rotation
        self.rotation_angle = 0
        self.old_distance = 0
        self.old_x = pos[0]
        self.best_distance = 9999

        self.cargo_rect = pygame.Rect(self.x + (PLAYER_IMG_WIDTH / 2) - 5, self.y + (PLAYER_IMG_HEIGHT / 2) + 6, 10, 10)

        self.is_collided = False
        self.is_dead = False

    def go_up(self):
        self.y_acc = -1

    def go_down(self):
        self.y_acc = 1

    def go_right(self):
        self.x_acc = 1
        self.rotation_angle += -5

    def go_left(self):
        self.x_acc = -1
        self.rotation_angle += 5

    def move(self):
        # y movement
        self.y_vel += self.y_acc + GRAVITY
        if self.y_vel < self.y_vel_up_boundary:
            self.y_vel = self.y_vel_up_boundary
        elif self.y_vel > self.y_vel_down_boundary:
            self.y_vel = self.y_vel_down_boundary
        if self.y_vel > 0:
            self.y_vel -= DRAG
        elif self.y_vel < 0:
            self.y_vel += DRAG
        self.y += self.y_vel
        self.y_acc = 0
        # x movement
        self.x_vel += self.x_acc
        if self.x_vel > 0:
            self.x_vel -= DRAG
        elif self.x_vel < 0:
            self.x_vel += DRAG
        if self.x_vel > 10:
            self.x_vel = 10
        elif self.x_vel < -10:
            self.x_vel = -10
        self.x += self.x_vel
        self.x_acc = 0
        # rotation
        if self.rotation_angle > 30:
            self.rotation_angle = 30
        elif self.rotation_angle < -30:
            self.rotation_angle = -30
        if self.rotation_angle > 0:
            self.rotation_angle -= 2
        elif self.rotation_angle < 0:
            self.rotation_angle += 2

    def draw(self, screen):
        self.cargo_rect = pygame.Rect(self.x + (PLAYER_IMG_WIDTH / 2) - 5, self.y + (PLAYER_IMG_HEIGHT / 2) + 6, 10, 10)
        rotated_img = pygame.transform.rotate(self.img, self.rotation_angle)
        new_rect = rotated_img.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        screen.blit(rotated_img, new_rect.topleft)

    def draw_rectangles(self, screen):
        cargo_rect = pygame.Rect(self.x + (PLAYER_IMG_WIDTH / 2) - 5, self.y + (PLAYER_IMG_HEIGHT / 2) + 6, 10, 10)
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.x, self.y, PLAYER_IMG_WIDTH, PLAYER_IMG_HEIGHT))
        pygame.draw.rect(screen, (0, 255, 0), cargo_rect)

    def draw_line_to_target(self, screen, target):
        pygame.draw.line(screen, (0, 0, 255), self.cargo_rect.center, target.center())

    def distance_to_target(self, target):
        return hypot(target.center()[0] - self.cargo_rect.center[0], target.center()[1] - self.cargo_rect.center[1])


class Target:

    def __init__(self):
        self.x = random.randint(30, SCREEN_WIDTH - 30)
        self.y = random.randint(30, SCREEN_HEIGHT - 30)
        self.width = 5
        self.color = (0, 255, 0)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.width)

    def center(self):
        return self.x + (self.width / 2), self.y + (self.width / 2)


def draw_screen(screen, drones, target, successfull, generation):
    screen.fill((0, 0, 0))
    for drone in drones:
        drone.draw_rectangles(screen=screen)
        drone.draw_line_to_target(screen=screen, target=target)
        drone.draw(screen=screen)

    target.draw(screen=screen)

    textsurface = myfont.render(f'Success: {successfull}', False, (255, 255, 255))
    screen.blit(textsurface, (0, 0))

    textsurface = myfont.render(f'GEN: {generation}', False, (255, 255, 255))
    screen.blit(textsurface, (0, 20))


GEN = 0
SUCCESS_NUMBERS = []
MAX_TIMES = []


def main(genomes, config):
    clock = pygame.time.Clock()

    global GEN
    GEN += 1

    nets = []
    ge = []
    drones = []
    target = Target()
    successfull_drones = 0
    first_collide = False

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        drone = Drone((SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        drone.old_distance = drone.distance_to_target(target)
        drones.append(drone)
        g.fitness = 0
        ge.append(g)

    # create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Title and icon
    pygame.display.set_caption("Drone - NEAT")

    TIME_LIMIT = 5 * 60  # 10 SECOND
    time_counter = 0
    # Game Loop
    running = True
    while running:
        clock.tick(60)
        time_counter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        if len(drones) == 0 or time_counter >= TIME_LIMIT:
            break

        for idx, drone in enumerate(drones):
            output = nets[idx].activate((drone.cargo_rect.center[0] - target.center()[0],
                                         drone.cargo_rect.center[1] - target.center()[1]))

            if output[0] > 0.5:
                drone.go_up()
            if output[0] < -0.5:
                drone.go_down()
            if output[1] > 0.5:
                drone.go_right()
            if output[1] < -0.5:
                drone.go_left()

            drone.move()

            if drone.x > SCREEN_WIDTH + 50 or drone.x < -50 or drone.y > SCREEN_HEIGHT + 50 or drone.y < -50:
                drone.is_dead = True
                ge[idx].fitness -= 1
            if drone.distance_to_target(target) < 5:
                drone.is_collided = True
                if not first_collide:
                    MAX_TIMES.append(time_counter / FPS)
                    first_collide = True
                successfull_drones += 1
                ge[idx].fitness += 10

            drone_new_distance = drone.distance_to_target(target)
            if drone.old_distance < drone_new_distance:
                ge[idx].fitness -= 0.1
            if drone.old_distance > drone_new_distance:
                ge[idx].fitness += 0.1
            drone.old_distance = drone_new_distance

            ge[idx].fitness -= 0.1  # for time

            if drone.is_dead or drone.is_collided:
                drones.pop(idx)
                nets.pop(idx)
                ge.pop(idx)

        draw_screen(screen=screen, drones=drones, target=target, successfull=successfull_drones, generation=GEN)

        pygame.display.update()

    global SUCCESS_NUMBERS
    SUCCESS_NUMBERS.append(successfull_drones)


def run(config_path, iteration_number):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, iteration_number)

    # plt.xlabel("GENERATIONS")
    # plt.ylabel("SUCCESFULL GENOMES")
    # plt.plot(np.arange(iteration_number), SUCCESS_NUMBERS)
    # plt.savefig("GENERATIONS vs SUCCESFULL GENOMES")
    # plt.show()
    #
    # plt.xlabel("GENERATIONS")
    # plt.ylabel("BEST TIMES")
    # plt.plot(np.arange(iteration_number), MAX_TIMES)
    # plt.savefig("GENERATIONS vs BEST TIMES")
    # plt.show()

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    print(local_dir)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    print(config_path)
    run(config_path, 50)
