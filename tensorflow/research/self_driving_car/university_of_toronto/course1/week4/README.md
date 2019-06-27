# Week 4. Vehicle Dynamic Modeling

## Overview

The first task for automating an driverless vehicle is to define a model for how the vehicle moves given steering, throttle and brake commands. This module progresses through a sequence of increasing fidelity physics-based models that are used to design vehicle controllers and motion planners that adhere to the limits of vehicle capabilities.

We'll look at modeling, both the evolution of the `kinematics`, that is, `positions and velocities`, and the `dynamics` or `forces and torques` of a car and how they connect.

## Learning Objectives

- Develop a kinematic bicycle model of a car with velocity and steering angle inputs
- Develop a dynamic bicycle models of a car with velocity and steering angle inputs
- Differentiate between models of tire forces
- Develop a model for actuation in a car, from pedal and steering wheel to tire forces

## Lesson 1: Kinematic Modeling in 2D

<p align=center>
    <img src=images/kinematic_vs_dynamic.png>
</p>

Generally, vehicle motion can be modeled either by considering the geometric constraint that defines its motion or by considering all of the forces and moments acting on a vehicle. The first case is known as `Kinematic Modeling`.

When we instead include knowledge of the forces and moments acting on the vehicle, we're performing `Dynamic Modeling`.

<p align=center>
    <img src=images/2D_kinematic_modeling.png>
</p>

We're now ready to start `kinematic modeling` of a simple robot. The robot's motion is constrained to move forward because its wheels point in this direction.

This constraint is called a `nonholonomic` constraint, which means that it `restricts the rate of change of the position` of our robot.

The `velocity` of the robot is defined by the `tangent vector` to its path.

Let's define the `orientation angle` of the robot as `Theta`.

<p align=center>
    <img src=images/simple_robot_motion_kinematics.png>
</p>

This model takes input the forward `velocity` in `rotation rate` and represents the robot using a vector of `three states`, the x and y position of the robot and it's heading.

## Lesson 2: The Kinematic Bicycle Model

<p align=center>
    <img src=images/bicycle_kinematic_model.png>
</p>

To analyze the kinematics of the bicycle model, we must select a reference point X, Y on the vehicle which can be placed at the `center of the rear axle`, the `center of the front axle`, or at the `center of gravity or cg`.

The `selection of the reference point` changes the kinematic equations that result, which in turn change the `controller designs` that we'll use.

<p align=center>
    <img src=images/rear_tire_reference_point.png>
</p>

Because of the no slip condition, we once again have that `Omega`, the `rotation rate` of the bicycle, is equal to the `velocity over the instantaneous center of rotation`, radius R.

<p align=center>
    <img src=images/front_tire_reference_point.png>
</p>

The bicycle kinematic model can be reformulated when the center
of the front axle is taken as the reference point x, y.

The last scenario is when the `desired point` is placed at the `center of gravity or center of mass` as shown in the right-hand figure.

<p align=center>
    <img src=images/center_reference_point.png>
</p>

Because of the no slip constraints we enforce on the front and rear wheels, the direction of motion at the cg is slightly different from the forward velocity direction in either wheel and from the heading of the bicycle.

This `difference` is called the `slip angle or side slip angle`, which we'll refer to as `Beta`, and is measured as `the angular difference between the velocity at the cg and the heading of the bicycle`.

Lastly, because of the no slip condition, we can compute the slip angle from the geometry of our bicycle model. Given `LR`, the `distance from the rear wheel to the cg`, the slip angle `Beta` is equal to the ratio of `LR over L times tan Delta`.

Finally, it is not usually possible to instantaneously change the steering angle of a vehicle from one extreme of its range to another, as is currently possible with our kinematic model.

Since `Delta` is an input that would be `selected by a controller`, there is no restriction on how quickly it can change which is somewhat unrealistic. Instead, our kinematic models can be formulated with four states: `x, y, Theta, and the steering angle Delta`.

If we assume we can `only control the rate of change of the steering angle Phi`, we can simply extend our model to `include Delta as a state` and `use the steering rate Phi` as our modified input.

Our kinematic bicycle model is now complete.

<p align=center>
    <img src=images/state_space_representation.png>
</p>

## Weekly Assignment

- Module 4 Programming Exercise: `Kinematic Bicycle Model`
  
  In this notebook, you will implement the kinematic bicycle model. The model accepts velocity and steering rate inputs and steps through the bicycle kinematic equations. Once the model is implemented, you will provide a set of inputs to drive the bicycle in a figure 8 trajectory.

- Module 4 Programming Exercise: `Longitudinal Vehicle Model`
  
  In this notebook, you will implement the forward longitudinal vehicle model. The model accepts throttle inputs and steps through the longitudinal dynamic equations. Once implemented, you will be given a set of inputs that drives over a small road slope to test your model.
