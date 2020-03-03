import numpy as np
import cv2
import copy
import random
import math

WINDOW_NAME = "Particle Filter"
DELAY_MSEC = 50

RADIUS_drone = 10
DRONE_COLOR = (0, 0, 255)  # RED
THICKNESS = -1

PARTICLES = 200 #options: 100, 200
THRESHOLD = 50
RADIUS_PARTICLE = 5
PARTICLE_COLOR = (0, 255, 255)

BORDER_COLOR_DRONE = (0, 255, 0) #GREEN
IMAGE_READING_WINDOW_SIZE = 25 #options: 25, 50
WINDOW_THICKNESS = 2

BORDER_COLOR_PARTICLES = (0, 255, 255)


def transfer_center_location(x, y, map):
    map_width = map.shape[0]
    map_height = map.shape[1]
    center_coordinates = {'y_coordinate': y + (map_height / 2),
                          'x_coordinate': x + (map_width / 2)
                          }
    return center_coordinates


def apply_negarive_exponential(x):
    return np.exp(-1 * x)

class simulator:
    def __init__(self, image):
        self.image = image
        self.image_array = np.array(image).transpose([1,0,2])
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.drone_position = self.set_initial_drone_position(self.width, self.height)
        self.particles = []

    def set_initial_drone_position(self, width, height):
        center_coordinates = {'x': int(np.random.uniform(low=THRESHOLD, high=width - THRESHOLD, size=None)),
                              'y': int(np.random.uniform(low=THRESHOLD, high=height - THRESHOLD, size=None))
                              }
        return center_coordinates

    def generate_particles(self):
        particles = []
        for i in range(PARTICLES):
            x_loc = np.random.randint(low= max(THRESHOLD, self.drone_position['x'] - 8 * THRESHOLD),
                                      high=min(self.drone_position['x'] + 8 * THRESHOLD, self.width-THRESHOLD),
                                      size=None, dtype=int)
            y_loc = np.random.randint(low=max(THRESHOLD, self.drone_position['y'] - 8 * THRESHOLD),
                                      high=min(self.drone_position['y'] + 8 * THRESHOLD, self.height - THRESHOLD),
                                      size=None, dtype=int)

            particle = {'x': x_loc, 'y': y_loc}
            particles.append(particle)
        self.particles = particles

    def move(self, tag='drone'):
        if tag=='drone':
            movement_vector = np.array([np.random.choice([-10, 10]), np.random.choice([-10, 10])])
            self.movement_vector = movement_vector

            noise_vector = np.random.normal(loc=0, scale=5, size=2)

            displacement = self.movement_vector + noise_vector
            displacement = displacement.astype(int)

            if(THRESHOLD<self.drone_position['x'] + displacement[0]< self.width-THRESHOLD) and \
                    (THRESHOLD<self.drone_position['y'] + displacement[1]< self.height-THRESHOLD):
                self.drone_position['x'] += displacement[0]
                self.drone_position['y'] += displacement[1]
        else:
            # print("self.movement_vector", self.movement_vector)
            for i in range(PARTICLES):
                    temp_dict = copy.deepcopy(self.particles[i])

                    noise_vector_particle = np.random.normal(loc=0, scale=5, size=2)
                    displacement_particle = self.movement_vector + noise_vector_particle
                    displacement_particle = displacement_particle.astype(int)

                    if (THRESHOLD <temp_dict['x'] + displacement_particle[0] < self.width-THRESHOLD) and \
                    (THRESHOLD < temp_dict['y'] + displacement_particle[1] < self.height-THRESHOLD):

                        temp_dict['x'] += displacement_particle[0]
                        temp_dict['y'] += displacement_particle[1]

                    self.particles[i] = temp_dict

            # print("*******")

    def update_map(self):
        temp_img = copy.copy(self.image)
        particles = self.particles

        circle_position = self.drone_position['x'], self.drone_position['y']
        cv2.circle(img=temp_img, center=circle_position, radius=RADIUS_drone, color=DRONE_COLOR,
                   thickness=THICKNESS)

        reference_img = self.generate_reference_img(temp_img)
        observation_imgs = self.generate_observation_imgs(temp_img, particles)
        weight_array_particles = self.assign_weight_particles(reference_img, observation_imgs)
        self.normalized_weight_array_particles= weight_array_particles

        self.move(tag='drone')
        self.particles = self.return_drawn_particles()
        self.move(tag='particles')

        for i, particle in enumerate(self.particles):
            # count_similar_particle = self.particles.count(particle) # for drawing larger particles
            particle_position = particle['x'], particle['y']
            cv2.circle(img=temp_img, center=particle_position,
                       radius=math.ceil(150*self.normalized_weight_array_particles[i]),
                       color=PARTICLE_COLOR, thickness=THICKNESS)  #+count_similar_particle//3

        # ***** experiments *****
        min = np.sqrt(np.square(self.particles[0]['x'] - self.drone_position['x']) +
                      np.square(self.particles[0]['y'] - self.drone_position['y']))
        for i in range(1, PARTICLES):
            place_holder = np.sqrt(np.square(self.particles[i]['x'] - self.drone_position['x']) +
                             np.square(self.particles[i]['y'] - self.drone_position['y']))
            if min > place_holder:
                min = place_holder
        # print("min_cluster", min)

        keep_centroid_distance = []
        for i in range(PARTICLES):
            place_holder = np.sqrt(np.square(self.particles[i]['x'] - self.drone_position['x']) +
                                   np.square(self.particles[i]['y'] - self.drone_position['y']))
            keep_centroid_distance.append(place_holder)
        mean_cluster = np.mean(keep_centroid_distance)
        # print("mean_cluster", mean_cluster)
        # print("***** *****")
        # ***** experiments *****

        return temp_img

    def generate_reference_img(self, temp_img):
        x_left = self.drone_position['x'] - int(IMAGE_READING_WINDOW_SIZE / 2)
        y_left = self.drone_position['y'] - int(IMAGE_READING_WINDOW_SIZE / 2)
        x_right = self.drone_position['x'] + int(IMAGE_READING_WINDOW_SIZE / 2)
        y_right = self.drone_position['y'] + int(IMAGE_READING_WINDOW_SIZE / 2)
        start_point = x_left, y_left
        end_point = x_right, y_right

        reference_img = self.image_array[x_left:x_right, y_left:y_right]
        self.reference_img = reference_img
        cv2.rectangle(temp_img, start_point, end_point, color=BORDER_COLOR_DRONE, thickness=WINDOW_THICKNESS)

        return reference_img


    def generate_observation_imgs(self, temp_img, particles):
        observation_imgs = []
        for particle in self.particles:
            x_left = particle['x'] - int(IMAGE_READING_WINDOW_SIZE / 2)
            y_left = particle['y'] - int(IMAGE_READING_WINDOW_SIZE / 2)
            x_right = particle['x'] + int(IMAGE_READING_WINDOW_SIZE / 2)
            y_right = particle['y'] + int(IMAGE_READING_WINDOW_SIZE / 2)

            start_point = x_left, y_left
            end_point = x_right, y_right

            observation_img = self.image_array[x_left:x_right, y_left:y_right,:]
            observation_imgs.append(observation_img)

            #cv2.rectangle(temp_img, start_point, end_point, color=BORDER_COLOR_PARTICLES, thickness=WINDOW_THICKNESS)

        observation_imgs = np.array(observation_imgs)
        self.observation_imgs = observation_imgs

        return observation_imgs

    def assign_weight_particles(self, reference_img, observation_imgs):
        msn_error_list = []
        for observation_img in observation_imgs:
            msn_error = np.sqrt((np.square(reference_img - observation_img))).mean()
            msn_error_list.append(msn_error)

        weight_array_particles = apply_negarive_exponential(np.array(msn_error_list))
        sum_weights = np.sum(weight_array_particles)
        normalized_weight_array_particles = weight_array_particles/sum_weights

        return normalized_weight_array_particles

    def weighted_choice(self, objects, weights):
        weights = np.array(weights, dtype=np.float64)
        sum_of_weights = weights.sum()
        # standardization:
        np.multiply(weights, 1 / sum_of_weights, weights)
        weights = weights.cumsum()
        x = random()
        for i in range(len(weights)):
            if x < weights[i]:
                return objects[i], weights[i]

    def return_drawn_particles(self):
        good_particles = []
        check_length = 0
        while check_length < PARTICLES:
            good_particle= np.random.choice(self.particles, p=self.normalized_weight_array_particles)
            good_particles.append(good_particle)
            check_length+=1
        return good_particles

def main():
    file_name = "BayMap.png" #options: BayMAp.png (8), MarioMap.png (4), CityMap.png (6)
    image = cv2.imread(file_name, 3)
    my_simulator = simulator(image)
    my_simulator.generate_particles()

    while(True):
        temp_image = my_simulator.update_map()
        cv2.imshow(WINDOW_NAME, temp_image)
        k = chr(cv2.waitKey(0))
        if k == 'a':
            continue
        if k == 's':
            cv2.imwrite('screenshot.png', temp_image)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
