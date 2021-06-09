Course EE3274 is a design project on building a mini autonomous car. The car will need to travel through a track painted with a black line, retrieve a cube and bring it back to the starting point. A camera was used to perceive the front of the car. It's a group project of three and I wrote the software.

## Image processing

OpenCV was used for image processing, a threshold was applied to turn the images into black and white. Then a contours finder was used to find the boundary of the black line. The power of the left and right motors were controlled based on the offset of the black line. Therefore the car can be steered and kept on track.

![output1](https://user-images.githubusercontent.com/80482978/119492648-38579380-bd57-11eb-9add-441d2b18067a.png)

The cube that was waiting to be picked was painted green. If there are a significant number of green pixels found in the image, the car will chase those pixels and grab the cube with a clamp. Then the car will try to return to the black line and go back to the starting point. Once it has reached the starting point, there should be no black line ahead and the cube will be released.

![output2](https://user-images.githubusercontent.com/80482978/119492659-3c83b100-bd57-11eb-8e17-ff068a1b5bdb.png)

## In action

https://user-images.githubusercontent.com/80482978/119492664-3e4d7480-bd57-11eb-9f36-288837dd0a15.mp4

The car was powered by four 18650 cells. A Raspberry Pi 3 B+ was used for image processing and generate PWM signals to control the motors.

![Final Product](https://user-images.githubusercontent.com/80482978/121418510-ede33300-c962-11eb-88d4-773fd579ce89.png)
