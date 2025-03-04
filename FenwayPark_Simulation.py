from vpython import *
import numpy as np

# Choose launch parameters (v0 in mph, angle in degrees for vertical motion, bearing in degrees for z-direction)
v0_mph = 110        # initial speed in mph
angle_deg = 30    # launch angle (vertical)
bearing_deg = 10   # bearing angle (horizontal, relative to x-axis)

# Set initial camera position to make the animation dope
initial_camera_pos = vector(100, 80, 502) # start in the red seat
initial_camera_axis = vector(-5, -10, -90)

# Define a simple function to convert degrees to radians
def rad(deg):
    return deg * pi / 180

# ---------------- VPython Scene Setup ----------------

scene = canvas(title="Leapfrog Simulation of a Home Run at Fenway Park", width=1000, height=600)

# Define the field and landmarks
field = box(pos=vector(155, -1, 180), size=vector(320, 1, 370), color=color.green)

base_size = 1.25  
monster_green = vector(0x4A/255, 0x77/255, 0x7A/255)

home_plate = box(pos=vector(0, 0.05, 0), size=vector(base_size, 0.1, base_size), color=color.white)
first_base = box(pos=vector(0, 0.05, 90), size=vector(base_size, 0.1, base_size), color=color.white)
second_base = box(pos=vector(90, 0.05, 90), size=vector(base_size, 0.1, base_size), color=color.white)
third_base = box(pos=vector(90, 0.05, 0), size=vector(base_size, 0.1, base_size), color=color.white)

mound = sphere(pos=vector(60, -10, 45), radius=12, color=color.orange)

green_monster = box(pos=vector(315, 18.5, 107.5), size=vector(10, 37.166, 221), color=monster_green)
cf_wall = box(pos=vector(282, 8.5, 288), size=vector(5, 17, 153.97), color=monster_green)
cf_wall.rotate(angle=59.32*180/pi, axis=vector(0,1,0))
cf_prot = box(pos=vector(240, 2.5, 350), size=vector(3, 5, 30), color=monster_green)
cf_prot.rotate(angle=75.68*180/pi, axis=vector(0,1,0))
rf_wall = box(pos=vector(130, 2.5, 350), size=vector(3, 5, 200), color=monster_green)
rf_wall.rotate(angle=120*180/pi, axis=vector(0,1,0))

center = vector(30, 0, 332)
radius = 30
angle_start = -pi/2
angle_end = 0
num_cylinders = 100
cylinder_height = 5
cylinder_radius = 1.5
for i in range(num_cylinders):
    angle_val = angle_start + (angle_end - angle_start) * (i / (num_cylinders - 1))
    x = center.x + radius * sin(angle_val)
    z = center.z + radius * cos(angle_val)
    pos_cyl = vector(x, 0, z)
    cylinder(pos=pos_cyl, axis=vector(0, 1, 0)*cylinder_height, radius=cylinder_radius, color=monster_green)

rf_wall = box(pos=vector(0, 2.5, 318), size=vector(3, 5, 30), color=monster_green)

# Foul lines
foul_line_y = curve(pos=[vector(0, 0, 0), vector(310, 0, 0)], radius=0.05, color=color.white)
foul_line_x = curve(pos=[vector(0, 0, 0), vector(0, 0, 90)], radius=0.05, color=color.white)

# Fisk and Pesky Poles
fisk_pole = cylinder(pos=vector(310, 0, 0), axis=vector(0, 60, 0), radius=1, color=color.yellow)
pesky_pole = cylinder(pos=vector(0, 0, 302), axis=vector(0, 60, 0), radius=1, color=color.yellow)

# Add the baseball (with a trail)
ball = sphere(pos=vector(0, 3, 0), radius=2, color=color.white, make_trail=True, trail_color=color.red)

# ---------------- Simulation Constants and Verlet (Leapfrog) Setup ----------------

g = 32.17                       # Gravity (ft/s²)
rho = 0.0740                    # Air density in lb/ft³
Cd = 0.33                       # Drag coefficient for a baseball
circumference = 9.125 / 12      # Baseball circumference (ft)
A = (circumference**2) / (4 * pi)  # Cross-sectional area (ft²)
m = 0.32                        # Mass (lbs)
dt = 0.01                       # Time step (s)
Fd_coeff = 0.5 * rho * Cd * A   # Drag force coefficient

# Convert speeds and angles
v0 = v0_mph * 5280 / 3600  # convert mph to ft/s
theta = rad(angle_deg)      # vertical angle
phi = rad(bearing_deg)      # bearing angle (horizontal)

# Compute velocity components
vx0 = v0 * cos(theta) * cos(phi)
vy0 = v0 * sin(theta)
vz0 = v0 * cos(theta) * sin(phi)

# Initialize position and velocity
pos = vector(0, 3, 0)
vel = vector(vx0, vy0, vz0)
home_run_printed = False

# Define camera pan final positions
pan_steps = 4/dt
final_camera_pos = vector(-60, 30, 300)
final_camera_axis = vector(100, -10, -80)
pan_counter = 0

# Create a label that displays the initial velocity and launch angle
initial_label = label(pos=vector(0, 3, 0),
                   text=f"Initial velocity: {v0_mph} mph\nLaunch angle: {angle_deg}°",
                   xoffset=0, yoffset=60,
                   height=10, border=4, font='sans',
                   box=True, opacity=0.8, color=color.white)
                   
# Max height label
max_height = ball.pos.y # Initialize max height with the ball's starting height
# Create a label to display the maximum height
max_height_label = label(pos=ball.pos,
                         text=f"Max Height: {max_height:.2f} ft",
                         xoffset=0, yoffset=20,
                         height=10, border=4, font='sans',
                         box=True, opacity=0.8, color=color.white)
wall_label = label(pos=vector(1000, 1000, 1000), text="", xoffset=20, yoffset=10,
                   height=10, border=4, font='sans', box=True, opacity=0.8, color=color.white)

def drag_force(v):
    # Expecting and returning a VPython vector
    return -Fd_coeff * mag(v) * v + vector(0, -m * g, 0)

class Particle3D:
    def __init__(self, mass, x, y, z, vx, vy, vz):
        self.mass = mass
        self.position = vector(x, y, z)
        self.velocity = vector(vx, vy, vz)
        self.radius = circumference / 2 / pi
        self.has_hit_wall = False
        self.coefficient_of_restitution = 0.8

    def leapfrog(self, force_func, dt):
        """Performs a single step of the Leapfrog integration method."""
        acceleration = force_func(self.velocity) / self.mass
        half_velocity = self.velocity + 0.5 * acceleration * dt
        self.position = self.position + half_velocity * dt
        new_acceleration = force_func(half_velocity) / self.mass
        self.velocity = half_velocity + 0.5 * new_acceleration * dt

    def check_wall_collision(self):
        """Check if the ball has collided with the Green Monster wall and handle the collision."""
        # Calculate wall boundaries
        monster_x = 310
        monster_height = 37.166
        monster_z_min = -3
        monster_z_max = 218
        # Check if ball is approaching wall from the front
        approaching_wall = self.velocity.x > 0

        # Check for collision with wall
        if (not self.has_hit_wall and
            approaching_wall and
            monster_x - 3 < self.position.x <= monster_x + 3 and
            monster_z_min < self.position.z < monster_z_max and  # Within z bounds
            self.position.y < monster_height):  # Below top of wall
            print(self.position.x, self.position.z, self.position.y)
            # Reflect velocity with energy loss
            self.velocity.x = -self.velocity.x * self.coefficient_of_restitution
            # Indicate collision
            self.has_hit_wall = True
            return True
        return False

    @property
    def x(self):
        return self.position.x

    @property
    def y(self):
        return self.position.y
    
    @property
    def z(self):
        return self.position.z

    @property
    def vx(self):
        return self.velocity.x

    @property
    def vy(self):
        return self.velocity.y
    
    @property
    def vz(self):
        return self.velocity.z

    @property
    def horizontal_distance(self):
        return sqrt(self.x**2 + self.z**2)
    
    @property
    def hit_ground(self):
        return self.y <= 0 and self.vy < 0

particle = Particle3D(m, ball.pos.x, ball.pos.y, ball.pos.z, vx0, vy0, vz0)

while not particle.hit_ground:
    rate(150)
    
    # ------------------ Physics Simulation ------------------
    particle.leapfrog(drag_force, dt)

    if particle.check_wall_collision():
        wall_label.text = f"Wall Distance: {particle.horizontal_distance:.2f} ft"
        wall_label.pos = vector(ball.pos.x, 0, ball.pos.z)  # Update position

    ball.pos = vector(particle.x, particle.y, particle.z)

    # ------------------ Camera Pan ------------------
    if pan_counter < pan_steps:
        t_fraction = pan_counter / pan_steps  # fraction of the pan completed
        # Interpolate camera position and axis
        scene.camera.pos = initial_camera_pos + t_fraction * (final_camera_pos - initial_camera_pos)
        scene.camera.axis = initial_camera_axis + t_fraction * (final_camera_axis - initial_camera_axis)
        pan_counter += 1
        
    # Update the max height and the label if the ball reaches a new high
    if ball.pos.y > max_height:
        max_height = ball.pos.y
        max_height_label.text = f"Max Height: {max_height:.2f} ft"
        # Optionally, update label position if you want it to follow the ball:
        max_height_label.pos = ball.pos
       
    # Home run detection
    if (not home_run_printed and particle.x >= (green_monster.pos.x - green_monster.size.x/2) and particle.y > green_monster.size.y):
        print("HOME RUN OVER THE GREEN MONSTAH!");
        home_run_printed = True

distance_label = label(pos=vector(particle.x, 0, particle.z),
                text=f"Distance Travelled: {particle.horizontal_distance:.2f} ft",
                xoffset=0, yoffset=80,
                height=10, border=4, font='sans',
                box=True, opacity=0.8, color=color.white)
                     
if (not home_run_printed):
    if particle.has_hit_wall:
        print("Off the Monstah!")
    else:
        print("Warning track power!")

print("Press 'c' to close the window")
scene.waitfor('keydown key=c')  # Wait for a key press to close the window