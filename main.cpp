/*
        Compile option:
        g++ -o main AIO.cpp -Wall -std=c++11 -pthread -lpigpio \
        -lrt -I/usr/local/include/raspicam -L/opt/vc/lib -lraspicam -lraspicam_cv \
        -lmmal -lmmal_core -lmmal_util -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect
*/
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <termios.h>
#include <pigpio.h>
#include "raspicam_cv.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>


#define SAVE_CAPTURED_IMAGE 0
#define SHOW_DEBUG_IMAGE 1


/* Camera settings */
#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 240
#define CAMERA_FRAMERATE 120
#define CAMERA_OFFSET -5


/* Line tracking variables */
#define X_START 80 /* define the area to look for line */
#define X_END 240
#define Y_START 120
#define Y_END 239
#define BLACK_THRESHOLD 100
#define NUM_OFFSET_SAMPLE 10 /* How many lines will be used to determine the offset */
#define HORIZONTAL_LINE_WIDTH_THRESHOLD 30
#define NORMAL_LINE_WIDTH_THRESHOLD 10


/* Cube variables */
#define ARM_X_START 140
#define ARM_X_END 180
#define ARM_Y_START 40
#define ARM_Y_END 90


/* Motor */
#define MOTOR_FULL_SPEED 120
#define MOTOR_MANUAL_FULL_SPEED 180
#define MOTOR_MAX_SPEED 255
#define MOTOR_MIN_SPEED 60
#define MOTOR_UTURN_SPEED 180
#define MOTOR_HTURN_SPEED 100 /* Heavy turn motor speed */
#define MOTOR_OFFSET_SCALE 0.25


/* Robotic arm */
#define SERVO_OPEN 1100
#define SERVO_CLOSE 1500
#define SERVO_STOP 0


/* Pin connections */
#define PIN_ENA 12
#define PIN_ENB 13
#define PIN_IN1 17
#define PIN_IN2 27
#define PIN_IN3 22
#define PIN_IN4 23
#define PIN_SERVO 26


/* Constants, no need to change */
#define MOTOR_LEFT 0
#define MOTOR_RIGHT 1
#define MOTOR_FORWARD 0
#define MOTOR_BACKWARD 1
#define MOTOR_TURN_LEFT 0
#define MOTOR_TURN_RIGHT 1
#define AUTO_DRIVING 0
#define MANUAL_DRIVING 1
#define COMMAND_FORWARD 1
#define COMMAND_BACKWARD 2
#define COMMAND_LEFT 3
#define COMMAND_RIGHT 4
#define COMMAND_ROTATE_LEFT 5
#define COMMAND_ROTATE_RIGHT 6
#define COMMAND_GRAB 7
#define COMMAND_RELEASE 8


using namespace cv;
using namespace std;


struct block {
        int offset;
        int width;
        
        block(int o, int w) : offset(o), width(w) {}


        bool operator < (const block& a) const {
                if(abs(offset) < abs(a.offset)) {
                        return true;
                }
                return false;
    }
};


bool run = true;


raspicam::RaspiCam_Cv camera;
Mat image, image_color, image_hsv, image_green, image_blue, image_bw, image_track, cube_pattern, temp_image;
CascadeClassifier cube_cascade;


int path_offset[IMAGE_HEIGHT], path_width[IMAGE_HEIGHT], path_start[IMAGE_HEIGHT], path_end[IMAGE_HEIGHT];
int path_offset_sample[NUM_OFFSET_SAMPLE], path_width_sample[NUM_OFFSET_SAMPLE];
int avg_offset, avg_width;
int heavy_turning = 0, operation_mode = 0, pending_command = 0, servo_status = 0, enable_pattern_search = 1;
float benchmark_fps;


pthread_t pthread_command;


void *human_command(void*);
void command_execution();
void image_capture();
void image_processing();
void qr_code_scan();
void path_finding();
void decision_making();
void heavy_turn();
void hardcode_uturn(int direction);
void print_result();
void setup();
void clean_up();
void camera_init();
void servo_init();
void set_servo(int angle);
void motor_init();
void set_motor_direction(int motor, int direction);
void set_motor_power(int motor, int power);
void stop(int signum);


int main() {
        int frame = 0, frame_in_second = 0;


        if (gpioInitialise() < 0) return -1;
        gpioSetSignalFunc(SIGINT, stop);
        
        if(!cube_cascade.load("cube_cascade.xml")) {
                cout << "Error loading cube cascade!" << endl;
                return -1;
        }
        
        cube_pattern = imread("pattern.jpg", 0);


        setup();
        
        image_track = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, Scalar(0, 0, 0));
        image_bw = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
        
        pthread_create(&pthread_command, NULL, human_command, NULL);
        
        set_servo(SERVO_OPEN);
        time_sleep(1.0);
        
        double start_time = cv::getTickCount();
        
        while(run) {
                image_capture();
                image_processing();
                //qr_code_scan();
                path_finding();
                if(operation_mode == AUTO_DRIVING) {
                        decision_making();
                } else {
                        command_execution();
                }
                print_result();
                
                if(SAVE_CAPTURED_IMAGE == 1) {
                        cv::imwrite("./image/image_" + to_string(frame++) + ".jpg", image_bw);
                }
                
                /* Calculate FPS */
                ++frame_in_second;
                double used_time = double (cv::getTickCount() - start_time) / double (cv::getTickFrequency()); /* In seconds */
                if(used_time > 1.0) {
                        benchmark_fps = (float) ((float) (frame_in_second) / used_time);
                        start_time = cv::getTickCount();
                        frame_in_second = 0;
                }
        }
        
        cout << "Cleaning up..." << endl;
        
        clean_up();


        return 0;
}


void command_execution() {
        if(pending_command == 0) {
                set_motor_power(MOTOR_LEFT, 0);
                set_motor_power(MOTOR_RIGHT, 0);
                return;
        }
        if(pending_command == COMMAND_FORWARD) {
                set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                set_motor_power(MOTOR_LEFT, MOTOR_MANUAL_FULL_SPEED);
                set_motor_power(MOTOR_RIGHT, MOTOR_MANUAL_FULL_SPEED);
        } else if(pending_command == COMMAND_BACKWARD) {
                set_motor_direction(MOTOR_LEFT, MOTOR_BACKWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_BACKWARD);
                set_motor_power(MOTOR_LEFT, MOTOR_MANUAL_FULL_SPEED);
                set_motor_power(MOTOR_RIGHT, MOTOR_MANUAL_FULL_SPEED);
        } else if(pending_command == COMMAND_LEFT) {
                set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                set_motor_power(MOTOR_LEFT, MOTOR_MIN_SPEED);
                set_motor_power(MOTOR_RIGHT, MOTOR_MANUAL_FULL_SPEED);
        } else if(pending_command == COMMAND_RIGHT) {
                set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                set_motor_power(MOTOR_LEFT, MOTOR_MANUAL_FULL_SPEED);
                set_motor_power(MOTOR_RIGHT, MOTOR_MIN_SPEED);
        } else if(pending_command == COMMAND_ROTATE_LEFT) {
                set_motor_direction(MOTOR_LEFT, MOTOR_BACKWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                set_motor_power(MOTOR_LEFT, MOTOR_MANUAL_FULL_SPEED);
                set_motor_power(MOTOR_RIGHT, MOTOR_MANUAL_FULL_SPEED);
        } else if(pending_command == COMMAND_ROTATE_RIGHT) {
                set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_BACKWARD);
                set_motor_power(MOTOR_LEFT, MOTOR_MANUAL_FULL_SPEED);
                set_motor_power(MOTOR_RIGHT, MOTOR_MANUAL_FULL_SPEED);
        } else if(pending_command == COMMAND_GRAB) {
                set_servo(SERVO_CLOSE);
        } else if(pending_command == COMMAND_RELEASE) {
                set_servo(SERVO_OPEN);
                time_sleep(0.25);
                set_servo(SERVO_STOP);
        }
        time_sleep(0.25);
        set_motor_power(MOTOR_LEFT, 0);
        set_motor_power(MOTOR_RIGHT, 0);
        pending_command = 0;
        if(operation_mode > 1) {
                operation_mode = AUTO_DRIVING;
        }
}


void image_capture() {
        camera.grab();
        camera.retrieve(temp_image);
        flip(temp_image, image_color, 1);
        cvtColor(image_color, image, COLOR_BGR2GRAY);
}


void image_processing() {
        image_track = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, Scalar(0, 0, 0));


        GaussianBlur(image, image_bw, Size(5, 5), 0, 0);
        image_bw = image > BLACK_THRESHOLD;


        /* Fill the black blocks with white */
        Mat image_bw_inverted;
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        
        bitwise_not(image_bw, image_bw_inverted);
        findContours(image_bw_inverted, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));


        for(unsigned int i = 0; i < contours.size(); i++) {
                bool edge_flag = false;
                for(unsigned int j = 0; j < contours[i].size(); j++) {
                        if(contours[i][j].x == 0 || contours[i][j].y == 0 || contours[i][j].x == IMAGE_WIDTH-1 || contours[i][j].y == IMAGE_HEIGHT-1) {
                                edge_flag = true;
                        }
                        if(contours[i][j].x > 120 && contours[i][j].x < 200 && contours[i][j].y > 80 && contours[i][j].y < 160) {
                                edge_flag = false;
                                break;
                        }
                }
                if(!edge_flag)
                        continue;


                drawContours(image_bw, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 0, Point());
        }


        /* Draw black blocks which on the edge */
        /*
        RNG rng(12345);
        for( int i = 0; i< contours.size(); i++ ) {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours(image_color, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }
        */


        /* Finding track */
        int x, y;
        int block_start, block_end, block_size, block_mid;
        int blue, green, red;
        for (y = Y_START; y < Y_END; y++) {
                vector<block> blocks;
                block_start = 0;
                block_end = -1;
                block_size = 0;
                block_mid = 0; // Experimental
                for (x = X_START; x < X_END; x++) {
                        if(image_bw.at<uint8_t>(y, x) < BLACK_THRESHOLD) {
                                if(block_start > block_end) {
                                        block_start = x;
                                }
                                block_end = x;
                                if( ( block_end - block_start ) > block_size) {
                                        block_size = block_end - block_start;
                                        block_mid = ( block_start + block_end ) / 2;
                                }
                        } else {
                                if(block_end > block_start + NORMAL_LINE_WIDTH_THRESHOLD) {
                                        blocks.push_back(block((( block_start + block_end ) / 2) - (IMAGE_WIDTH / 2), block_end - block_start));
                                }
                                block_start = x;
                        }
                }
                if(block_end > block_start + NORMAL_LINE_WIDTH_THRESHOLD) {
                        blocks.push_back(block((( block_start + block_end ) / 2) - (IMAGE_WIDTH / 2), block_end - block_start));
                }
                
                //path_offset[y] = block_mid - (IMAGE_WIDTH / 2);
                //path_width[y] = block_size;
                
                if(!blocks.empty()) {
                        sort(blocks.begin(), blocks.end());
                        path_offset[y] = blocks.front().offset;
                        path_width[y] = blocks.front().width;
                } else {
                        path_offset[y] = 0;
                        path_width[y] = 0;
                }
                
                for (x = X_START; x < X_END; x++) {
                        int mid = path_offset[y] + (IMAGE_WIDTH / 2);
                        int start = mid - (block_size / 2);
                        int end = mid + (block_size / 2);
                        blue = 255;
                        green = 255;
                        red = 255;
                        if(x == mid) {
                                blue = 0;
                                green = 0;
                                red = 255;
                        } else if(x == start || x == end) {
                                blue = 0;
                                green = 0;
                                red = 0;
                        }
                        image_track.at<Vec3b>(y, x) = Vec3b(blue, green, red);
                }
        }
}


void qr_code_scan() {
        
}


int compare(const void *a, const void *b) {
        return (*(int*)a - *(int*)b);
}


void path_finding() {
        avg_offset = 0, avg_width = 0;


        for(int i = 0; i < NUM_OFFSET_SAMPLE; i++) {
                path_offset_sample[i] = path_offset[Y_START + i];
                path_width_sample[i] = path_width[Y_START + i];
        }


        //qsort(path_offset_sample, NUM_OFFSET_SAMPLE, sizeof(int), compare);
        //qsort(path_width_sample, NUM_OFFSET_SAMPLE, sizeof(int), compare);


        for(int i = 1; i < NUM_OFFSET_SAMPLE - 1; i++) {
                avg_offset += path_offset_sample[i];
                avg_width += path_width_sample[i];
        }


        avg_offset = avg_offset / (NUM_OFFSET_SAMPLE - 2) + CAMERA_OFFSET;
        avg_width = avg_width / (NUM_OFFSET_SAMPLE - 2);
}


bool green_cube_matching() {
        inRange(image_hsv, Scalar(40, 100, 50), Scalar(80, 255, 255), image_green);


        int cube_width = 0, cube_height = 0, cube_x = 0, cube_y = 0;
        int block_start, block_end, block_size, block_mid;
        for(int y = 0; y < IMAGE_HEIGHT; y++) {
                block_start = 0;
                block_end = -1;
                block_size = 0;
                for(int x = 0; x < IMAGE_WIDTH; x++) {
                        if(image_green.at<uint8_t>(y, x) > BLACK_THRESHOLD) {
                                if(block_start > block_end) {
                                        block_start = x;
                                }
                                block_end = x;
                                if( ( block_end - block_start ) > block_size) {
                                        block_size = block_end - block_start;
                                        block_mid = ( block_start + block_end ) / 2;
                                }
                        } else {
                                block_start = x;
                        }
                }
                if(block_size > 10) {
                        cube_height++;
                        cube_width += block_size;
                        cube_x += block_mid;
                        cube_y += y;
                }
        }


        if(cube_height > 10) {
                cube_width = cube_width / cube_height;
                cube_x = cube_x / cube_height;
                cube_y = cube_y / cube_height;
                
                cube_x = cube_x > IMAGE_WIDTH-1 ? IMAGE_WIDTH-1 : cube_x;
                cube_x = cube_x < 0 ? 0 : cube_x;
                cube_y = cube_y > IMAGE_HEIGHT-1 ? IMAGE_HEIGHT-1 : cube_y;
                cube_y = cube_y < 0 ? 0 : cube_y;
        
                circle(image, Point(cube_x, cube_y), 20, Scalar(0, 0, 0), 5, 8, 0);
                
                set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                
                if(cube_y > ARM_Y_START && cube_y < ARM_Y_END) {
                        /* Stop and grab item */
                        set_motor_power(MOTOR_LEFT, 0);
                        set_motor_power(MOTOR_RIGHT, 0);
                        set_servo(SERVO_CLOSE);
                        time_sleep(1.0);
                        
                        /* Backing up */
                        set_motor_direction(MOTOR_LEFT, MOTOR_BACKWARD);
                        set_motor_direction(MOTOR_RIGHT, MOTOR_BACKWARD);
                        set_motor_power(MOTOR_LEFT, 160);
                        set_motor_power(MOTOR_RIGHT, 160);
                        time_sleep(2.0);
                        
                        /* Disable pattern search and u turn */
                        //enable_pattern_search = 0;
                        cout << "Item grabbed." << endl;
                        
                        /* Turn left */
                        set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                        set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                        set_motor_power(MOTOR_LEFT, 0);
                        set_motor_power(MOTOR_RIGHT, 140);
                        time_sleep(1);
                        
                        do {
                                image_capture();
                                image_processing();
                                path_finding();
                                print_result();
                                time_sleep(0.1);
                        } while((abs(avg_offset) > 60 || avg_width < 10) && run);
                } else {
                        int motor_power = MOTOR_FULL_SPEED + (int) ((double) (cube_x - IMAGE_WIDTH / 2) / 0.5);
                        motor_power = motor_power > MOTOR_MAX_SPEED ? MOTOR_MAX_SPEED : motor_power;
                        motor_power = motor_power < MOTOR_MIN_SPEED ? MOTOR_MIN_SPEED : motor_power;
                        set_motor_power(MOTOR_LEFT, motor_power);
                        
                        motor_power = MOTOR_FULL_SPEED - (int) ((double) (cube_x - IMAGE_WIDTH / 2) / 0.5);
                        motor_power = motor_power > MOTOR_MAX_SPEED ? MOTOR_MAX_SPEED : motor_power;
                        motor_power = motor_power < MOTOR_MIN_SPEED ? MOTOR_MIN_SPEED : motor_power;
                        set_motor_power(MOTOR_RIGHT, motor_power);
                }
                return true;
        }
        return false;
}


bool blue_bed_matching() {
        inRange(image_hsv, Scalar(100, 100, 100), Scalar(140, 255, 255), image_blue);


        int cube_width = 0, cube_height = 0, cube_x = 0, cube_y = 0;
        int block_start, block_end, block_size, block_mid;
        for(int y = 0; y < IMAGE_HEIGHT; y++) {
                block_start = 0;
                block_end = -1;
                block_size = 0;
                for(int x = 0; x < IMAGE_WIDTH; x++) {
                        if(image_blue.at<uint8_t>(y, x) > BLACK_THRESHOLD) {
                                if(block_start > block_end) {
                                        block_start = x;
                                }
                                block_end = x;
                                if( ( block_end - block_start ) > block_size) {
                                        block_size = block_end - block_start;
                                        block_mid = ( block_start + block_end ) / 2;
                                }
                        } else {
                                block_start = x;
                        }
                }
                if(block_size > 20) {
                        cube_height++;
                        cube_width += block_size;
                        cube_x += block_mid;
                        cube_y += y;
                }
        }


        if(cube_height > 20) {
                cube_width = cube_width / cube_height;
                cube_x = cube_x / cube_height;
                cube_y = cube_y / cube_height;
                
                cube_x = cube_x > IMAGE_WIDTH-1 ? IMAGE_WIDTH-1 : cube_x;
                cube_x = cube_x < 0 ? 0 : cube_x;
                cube_y = cube_y > IMAGE_HEIGHT-1 ? IMAGE_HEIGHT-1 : cube_y;
                cube_y = cube_y < 0 ? 0 : cube_y;
        
                circle(image, Point(cube_x, cube_y), 20, Scalar(255, 0, 0), 5, 8, 0);
                
                cout << "Blue block found." << endl;
                
                if(servo_status == SERVO_CLOSE) {
                        hardcode_uturn(-1); // Turn left
                } else {
                        hardcode_uturn(1); // Turn right
                }
                
                return true;
        }
        return false;
}


bool turning_point_matching() {
        bool point_matched = true;
        for(int y = Y_START; y < Y_START + 40; y++) {
                if(abs(path_offset[y]) > 40 || path_width[y] > 40) {
                        point_matched = false;
                }
        }
        
        for(int y = Y_START + 40; y < Y_START + 80; y++) {
                if(path_width[y] > 10) {
                        point_matched = false;
                }
        }
        
        if(point_matched) {
                cout << "Turning point found." << endl;
                
                if(servo_status == SERVO_CLOSE) {
                        hardcode_uturn(-1); // Turn left
                } else {
                        hardcode_uturn(1); // Turn right
                }
                
                return true;
        }
        
        return false;
}


void decision_making() {
        /* Color matching */
        if(enable_pattern_search == 1) {
                cvtColor(image_color, image_hsv, COLOR_BGR2HSV);
                if(servo_status != SERVO_CLOSE) {
                        if(green_cube_matching()) {
                                return;
                        }
                }
                if(turning_point_matching()) {
                        return;
                }
                /*
                if(blue_bed_matching()) {
                        return;
                }
                */
        }
        
        if(heavy_turning != 0) {
                if(abs(avg_offset) < 80) {
                        heavy_turning = 0;
                        return;
                }
                time_sleep(0.1);
                heavy_turn();
        } else {
                if(avg_width > HORIZONTAL_LINE_WIDTH_THRESHOLD && avg_width < X_END - X_START) {
                        heavy_turning = avg_offset;
                        printf("Decision making: Heavy turning (offset: %d, width: %d)\n", avg_offset, avg_width);
                        set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                        set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                        //set_motor_power(MOTOR_LEFT, MOTOR_FULL_SPEED);
                        //set_motor_power(MOTOR_RIGHT, MOTOR_FULL_SPEED);
                        time_sleep(0.1);
                        heavy_turn();
                        return;
                }
                int avg_lead_width = 0;
                for(int i = Y_START + NUM_OFFSET_SAMPLE; i < Y_START + NUM_OFFSET_SAMPLE * 2; i++) {
                        avg_lead_width += path_width[i];
                }
                avg_lead_width = avg_lead_width / NUM_OFFSET_SAMPLE;


                set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
                set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);
                
                int motor_power = MOTOR_FULL_SPEED + (int) ((double) avg_offset / MOTOR_OFFSET_SCALE);
                motor_power = motor_power > MOTOR_MAX_SPEED ? MOTOR_MAX_SPEED : motor_power;
                motor_power = motor_power < MOTOR_MIN_SPEED ? MOTOR_MIN_SPEED : motor_power;
                set_motor_power(MOTOR_LEFT, motor_power);
                
                //cout << "Decision making: Left " << motor_power;
                
                motor_power = MOTOR_FULL_SPEED - (int) ((double) avg_offset / MOTOR_OFFSET_SCALE);
                motor_power = motor_power > MOTOR_MAX_SPEED ? MOTOR_MAX_SPEED : motor_power;
                motor_power = motor_power < MOTOR_MIN_SPEED ? MOTOR_MIN_SPEED : motor_power;
                set_motor_power(MOTOR_RIGHT, motor_power);
                
                //cout << " Right " << motor_power << endl;
        }
}


void heavy_turn() {
        int temp_motor_1, temp_motor_2, motor_power;
        if(heavy_turning > 0) { /* Turning right */
                temp_motor_1 = MOTOR_LEFT;
                temp_motor_2 = MOTOR_RIGHT;
        } else { /* Turning left */
                temp_motor_2 = MOTOR_LEFT;
                temp_motor_1 = MOTOR_RIGHT;
        }
        //motor_power = abs(heavy_turning) > 10000 ? MOTOR_UTURN_SPEED : MOTOR_HTURN_SPEED;
        motor_power = MOTOR_HTURN_SPEED;
        set_motor_direction(temp_motor_1, MOTOR_FORWARD);
        set_motor_direction(temp_motor_2, MOTOR_BACKWARD);
        set_motor_power(MOTOR_LEFT, motor_power);
        set_motor_power(MOTOR_RIGHT, motor_power);
}


void hardcode_uturn(int direction) {
        int temp_motor_1, temp_motor_2, motor_power;
        if(direction > 0) { /* Turning right */
                temp_motor_1 = MOTOR_LEFT;
                temp_motor_2 = MOTOR_RIGHT;
                motor_power = 120;
                cout << "[DEBUG][hardcode_uturn] Turning right" << endl;
        } else { /* Turning left */
                temp_motor_2 = MOTOR_LEFT;
                temp_motor_1 = MOTOR_RIGHT;
                motor_power = 100;
                cout << "[DEBUG][hardcode_uturn] Turning left" << endl;
        }
        /* Slight turn first */
        set_motor_direction(temp_motor_1, MOTOR_FORWARD);
        set_motor_direction(temp_motor_2, MOTOR_FORWARD);
        /*
        set_motor_power(temp_motor_1, motor_power);
        set_motor_power(temp_motor_2, 0);
        time_sleep(1.0);
        */
        
        /* Heavy turn now */
        /*
        set_motor_direction(temp_motor_1, MOTOR_FORWARD);
        set_motor_direction(temp_motor_2, MOTOR_BACKWARD);
        set_motor_power(temp_motor_1, motor_power);
        set_motor_power(temp_motor_2, motor_power);
        time_sleep(0.25);
        */
        
        /* Back to normal */
        if(direction > 0) { /* Turning right */
                set_motor_power(temp_motor_1, motor_power);
                set_motor_power(temp_motor_2, 0);
                time_sleep(1.0);
                do {
                        image_capture();
                        image_processing();
                        path_finding();
                        print_result();
                        time_sleep(0.1);
                } while((abs(avg_offset) > 60 || avg_width < 10) && run);
        } else {
                set_motor_direction(temp_motor_1, MOTOR_FORWARD);
                set_motor_direction(temp_motor_2, MOTOR_BACKWARD);
                set_motor_power(temp_motor_1, motor_power);
                set_motor_power(temp_motor_2, motor_power);
                time_sleep(0.5);
                do {
                        image_capture();
                        image_processing();
                        path_finding();
                        print_result();
                        time_sleep(0.05);
                } while((abs(avg_offset) > 60 || avg_width < 10) && run);
        }
}


void print_result() {
        char info[100];
        sprintf(info, "FPS: %d Offset: %d (%d)", (int) benchmark_fps, avg_offset, avg_width);
        putText(image_track, info, Point(2, 15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0, 8, false);
        rectangle(image, Point(ARM_X_START, ARM_Y_START), Point(ARM_X_END, ARM_Y_END), Scalar(255, 255, 255), 1, 8, 0);
        cv::imshow("Captured image", image);
        cv::imshow("Image in B&W", image_bw);
        cv::imshow("Track", image_track);
        if(SHOW_DEBUG_IMAGE == 1 && enable_pattern_search == 1) {
                if(image_green.rows > 0) {
                        cv::imshow("Image Green Channel", image_green);
                }
                if(image_blue.rows > 0) {
                        cv::imshow("Image Blue Channel", image_blue);
                }
        }
        cv::waitKey(1);
}


/* Remote controls */
int getch(void) {
        int ch;
        struct termios oldt;
        struct termios newt;
        tcgetattr(STDIN_FILENO, &oldt); /*store old settings */
        newt = oldt; /* copy old settings to new settings */
        newt.c_lflag &= ~(ICANON | ECHO); /* make one change to old settings in new settings */
        tcsetattr(STDIN_FILENO, TCSANOW, &newt); /*apply the new settings immediatly */
        ch = getchar(); /* standard getchar call */
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt); /*reapply the old settings */
        return ch; /*return received char */
}


void *human_command(void*) {
        printf("press *m* to enter manual driving mode\n");
        printf("press *n* to return to auto driving mode\n");
        printf("press *w/a/s/d* to move your car\n");
        printf("press *q/e* to rotate your car\n");
        printf("press *g* to grab item\n");
        printf("press *r* to release item\n");
        printf("press *p* to toggle color searching\n");


        int c;
        while(run) {
                c = getch();
                if(tolower(c) == (int) 'm') {
                        pending_command = 0;
                        operation_mode = MANUAL_DRIVING;
                        printf("Entering manual driving mode.\n");
                        continue;
                } else if(tolower(c) == (int) 'n') {
                        operation_mode = AUTO_DRIVING;
                        printf("Going back to auto driving mode.\n");
                        continue;
                } else if(tolower(c) == (int) 'p') {
                        enable_pattern_search = !enable_pattern_search;
                        printf("Color searching mode: %d.\n", enable_pattern_search);
                        continue;
                }
                
                if(tolower(c) == (int) 'w') {
                        pending_command = COMMAND_FORWARD;
                        printf("Moving forward.\n");
                } else if(tolower(c) == (int) 'a') {
                        pending_command = COMMAND_LEFT;
                        printf("Moving left.\n");
                } else if(tolower(c) == (int) 's') {
                        pending_command = COMMAND_BACKWARD;
                        printf("Moving back.\n");
                } else if(tolower(c) == (int) 'd') {
                        pending_command = COMMAND_RIGHT;
                        printf("Moving right.\n");
                } else if(tolower(c) == (int) 'q') {
                        pending_command = COMMAND_ROTATE_LEFT;
                        printf("Turning left.\n");
                } else if(tolower(c) == (int) 'e') {
                        pending_command = COMMAND_ROTATE_RIGHT;
                        printf("Turning right.\n");
                } else if(tolower(c) == (int) 'g') {
                        pending_command = COMMAND_GRAB;
                        printf("Grabing.\n");
                } else if(tolower(c) == (int) 'r') {
                        pending_command = COMMAND_RELEASE;
                        printf("Releasing.\n");
                }
                
                if(operation_mode != MANUAL_DRIVING) {
                        operation_mode = 2;
                }
        }
        
        pthread_exit(NULL);
}


void setup() {
        camera_init();
        servo_init();
        motor_init();


        set_servo(SERVO_STOP);


        set_motor_direction(MOTOR_LEFT, MOTOR_FORWARD);
        set_motor_direction(MOTOR_RIGHT, MOTOR_FORWARD);


        set_motor_power(MOTOR_LEFT, 0);
        set_motor_power(MOTOR_RIGHT, 0);
}


void clean_up() {
        gpioWrite(PIN_IN1, 0);
        gpioWrite(PIN_IN2, 0);
        gpioWrite(PIN_IN3, 0);
        gpioWrite(PIN_IN4, 0);
        set_motor_power(MOTOR_LEFT, 0);
        set_motor_power(MOTOR_RIGHT, 0);
        set_servo(SERVO_STOP);
        gpioTerminate();
        camera.release();
        printf("Cleaning completed.\n");
        //pthread_kill(pthread_command, 15); // SIGTERM
        //pthread_exit(NULL);
}


/* Camera functions */
void camera_init() {
        camera.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
        camera.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
        camera.set(CV_CAP_PROP_BRIGHTNESS, 50);
        camera.set(CV_CAP_PROP_CONTRAST, 50);
        camera.set(CV_CAP_PROP_SATURATION, 50);
        camera.set(CV_CAP_PROP_GAIN, 50);
        camera.set(CV_CAP_PROP_FORMAT, CV_8UC3);
        camera.set(CV_CAP_PROP_FPS, CAMERA_FRAMERATE);
        if(!camera.open()) {
                cerr << "Error opening camera." << endl;
                run = false;
        }
        cout << "Connected to camera #" << camera.getId() << endl;
}


/* Servo motor functions */
void servo_init() {
        gpioSetMode(PIN_SERVO, PI_OUTPUT);
}


void set_servo(int angle) {
        gpioServo(PIN_SERVO, angle);
        servo_status = angle;
}


/* Motor functions */
void motor_init() {
        gpioSetMode(PIN_ENA, PI_OUTPUT);
        gpioSetMode(PIN_ENB, PI_OUTPUT);
        gpioSetMode(PIN_IN1, PI_OUTPUT);
        gpioSetMode(PIN_IN2, PI_OUTPUT);
        gpioSetMode(PIN_IN3, PI_OUTPUT);
        gpioSetMode(PIN_IN4, PI_OUTPUT);
}


void set_motor_direction(int motor, int direction) {
        int tmp_pin_1, tmp_pin_2;
        if(motor == MOTOR_LEFT) {
                tmp_pin_1 = PIN_IN1;
                tmp_pin_2 = PIN_IN2;
        } else {
                tmp_pin_1 = PIN_IN3;
                tmp_pin_2 = PIN_IN4;
        }
        if(direction == MOTOR_FORWARD) {
                gpioWrite(tmp_pin_1, 1);
                gpioWrite(tmp_pin_2, 0);
        } else {
                gpioWrite(tmp_pin_1, 0);
                gpioWrite(tmp_pin_2, 1);
        }
}


void set_motor_power(int motor, int power) {
        power = power > 255 ? 255 : power;
        power = power < 0 ? 0 : power;
        if(motor == MOTOR_LEFT) {
                gpioPWM(PIN_ENA, power);
        } else {
                gpioPWM(PIN_ENB, power);
        }
}


void stop(int signum) {
        run = false;
}
