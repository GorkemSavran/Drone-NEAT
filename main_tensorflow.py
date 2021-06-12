import pygame
import random
import neat
from math import hypot
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

KERAS_MODEL_SHAPE = [6, 2]  # 6 input 6 hidden 2 output

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
FPS = 30

PLAYER_IMG = pygame.image.load("assets/drone.png")
PLAYER_IMG_HEIGHT = PLAYER_IMG.get_height()
PLAYER_IMG_WIDTH = PLAYER_IMG.get_width()

GRAVITY = 0.5
DRAG = 0.5

# Initialize pygame
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)


def create_model():
    model = Sequential()
    # model.add(Dense(6, activation="relu", ))
    model.add(Dense(2, activation="tanh", input_dim=6))
    return model


def selection_roulette_wheel(genes, scores):
    """
    pi = fi / sum(fi)
    """
    probes = []
    selected = []
    sum_of_scores = np.sum(scores)
    # Find probes
    for idx, score in enumerate(scores):
        # Normalize probes
        probes.append((score / sum_of_scores))

    for i in range(int(len(genes) / 2)):
        turn_wheel = np.random.rand(1)[0]
        counter = probes[0]
        idx = 0
        while counter < turn_wheel:
            counter += probes[idx + 1]
            idx += 1
        selected.append(genes[idx])
    return selected


def crossover_single_point(p1, p2, p_crossover=1.0):
    # Single point crossover
    # Choosing a random crossover point
    # default probability of crossover is 1.0
    c1 = p1.copy()
    c2 = p2.copy()
    if np.random.rand(1)[0] < p_crossover:
        crossover_point = np.random.randint(1, len(p1) - 1)
        c1 = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
        c2 = np.concatenate((p2[:crossover_point], p1[crossover_point:]))
    return c1, c2



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
        self.y_vel_up_boundary = -15
        self.y_vel_down_boundary = 15
        # acceleration
        self.y_acc = 0
        self.x_acc = 0
        # rotation
        self.rotation_angle = 0
        self.old_distance = 0
        self.old_x = pos[0]
        self.best_distance = 9999

        self.fitness_score = 0

        self.cargo_rect = pygame.Rect(self.x + (PLAYER_IMG_WIDTH / 2) - 5, self.y + (PLAYER_IMG_HEIGHT / 2) + 6, 10, 10)

        self.is_collided = False
        self.is_dead = False

    def load_brain(self, genome):
        self.genome = genome
        new_weights = []
        for i in range(len(KERAS_MODEL_SHAPE) - 1):
            n = KERAS_MODEL_SHAPE[i] * KERAS_MODEL_SHAPE[i + 1]
            new_weights.append(np.array(genome[:n]).reshape((KERAS_MODEL_SHAPE[i], KERAS_MODEL_SHAPE[i + 1])))
            genome = genome[n:]
        self.model = create_model()
        model_weights = self.model.get_weights()
        model_weights[::2] = new_weights
        self.model.set_weights(model_weights)
        self.model.compile()

    def brain_action(self, inputs):
        return self.model.predict(inputs)

    def go_up(self):
        self.y_acc = -1

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
        # drone.draw(screen=screen)

    target.draw(screen=screen)

    textsurface = myfont.render(f'Success: {successfull}', False, (255, 255, 255))
    screen.blit(textsurface, (0, 0))

    textsurface = myfont.render(f'GEN: {generation}', False, (255, 255, 255))
    screen.blit(textsurface, (0, 20))


GEN = 0


def main(genomes):
    clock = pygame.time.Clock()

    global GEN
    GEN += 1

    drones = []
    done_drones = []
    target = Target()
    successfull_drones = 0
    # drone = Drone()

    # rand_x = random.randint(10, SCREEN_WIDTH - 10)
    # rand_y = random.randint(10, SCREEN_HEIGHT - 10)
    for genome in genomes:
        drone = Drone((SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        drone.load_brain(genome)
        drone.old_distance = drone.distance_to_target(target)
        drones.append(drone)

    # create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Title and icon
    pygame.display.set_caption("Drone - NEAT")

    TIME_LIMIT = 5 * FPS  # SECOND
    time_counter = 0
    # Game Loop
    running = True
    while running:
        clock.tick(FPS)
        time_counter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        if len(drones) == 0:
            break

        if time_counter >= TIME_LIMIT:
            # sum_of_distances = 0
            # for drone in drones:
            #     sum_of_distances += drone.best_distance
            # average_distance = sum_of_distances / len(drones)
            # for idx, drone in enumerate(drones):
            #     print((average_distance - drone.best_distance) / 10)
            #     ge[idx].fitness += (average_distance - drone.best_distance) / 10
            break

        for idx, drone in enumerate(drones):
            inputs = np.array([drone.cargo_rect.center[0], drone.cargo_rect.center[1],
                               target.center()[0], target.center()[1],
                               drone.x_vel, drone.y_vel])

            output = drone.brain_action(inputs.reshape((-1, 6)))

            # print(output)
            if output[0][0] > 0:
                drone.go_up()
            if output[0][1] > 0:
                drone.go_right()
            if output[0][1] < 0:
                drone.go_left()

            drone.move()

            if drone.x > SCREEN_WIDTH + 20 or drone.x < -20 or drone.y > SCREEN_HEIGHT + 20 or drone.y < -20:
                drone.is_dead = True
                drone.fitness_score -= 1
            if drone.distance_to_target(target) < 15:
                drone.is_collided = True
                successfull_drones += 1
                drone.fitness_score += 10

            # point = drone.distance_to_target(target) / 10000
            # print(point)
            # ge[idx].fitness -= point
            # if drone.x == drone.old_x:
            #     ge[idx].fitness -= 1
            # drone.old_x = drone.x

            drone_new_distance = drone.distance_to_target(target)
            if drone.old_distance < drone_new_distance:
                drone.fitness_score -= 0.1
            if drone.old_distance > drone_new_distance:
                drone.fitness_score += 0.1
            drone.old_distance = drone_new_distance
            # if drone.best_distance > drone_new_distance:
            #     drone.best_distance = drone_new_distance
            # ge[idx].fitness -= 0.1  # for time

            if drone.is_dead or drone.is_collided:
                done_drones.append(drones.pop(idx))

        draw_screen(screen=screen, drones=drones, target=target, successfull=successfull_drones, generation=GEN)

        pygame.display.update()

    for idx, drone in enumerate(drones):
        done_drones.append(drones.pop(idx))

    drone_genomes = []
    drone_scores = []
    for drone in done_drones:
        drone_genomes.append(drone.genome)
        drone_scores.append(drone.fitness_score)

    return drone_genomes, drone_scores


def initialize_genomes(population_size):
    genomes = []
    for i in range(population_size):
        model = create_model()
        weights = np.array(model.get_weights())
        # print(weights)
        weights = weights[::2]
        weights_1d = np.concatenate([weight.reshape(-1) for weight in weights])
        genomes.append(weights_1d)
    return genomes


# def load_brain(genome):
#     new_weights = []
#     for i in range(len(KERAS_MODEL_SHAPE) - 1):
#         n = KERAS_MODEL_SHAPE[i] * KERAS_MODEL_SHAPE[i + 1]
#         new_weights.append(np.array(genome[:n]).reshape((KERAS_MODEL_SHAPE[i], KERAS_MODEL_SHAPE[i + 1])))
#         genome = genome[n:]
#     model = create_model()
#     model_weights = model.get_weights()
#     model_weights[::2] = new_weights
#     model.set_weights(model_weights)
#     return model


def run(iteration):
    genomes = initialize_genomes(4)
    # model = load_brain(genomes[0])
    # x = np.array([632, 393, 932.5, 526.5, 0, 0])
    # x = x.reshape((-1, 6))
    # print(model.predict(x))
    for i in range(iteration):
        drone_genomes, drone_scores = main(genomes)
        print("Drone Genomes: ", len(drone_genomes))
        # print("Drone Scores: ", drone_scores)
        selected_parents = selection_roulette_wheel(drone_genomes, drone_scores)
        new_genomes = []
        len_selected_parents = len(selected_parents)
        for idx in range(len_selected_parents):
            c1, c2 = crossover_single_point(selected_parents[random.randint(0, len_selected_parents-1)], selected_parents[random.randint(0, len_selected_parents-1)])
            new_genomes.append(c1)
            new_genomes.append(c2)
        genomes = new_genomes
        print(len(genomes))

if __name__ == "__main__":
    run(50)
