//   #include<Arduino.h>
//   #include<ESP32Servo.h>
//   #include <ESP32Time.h>
#include "commands.h"

#include<Arduino.h>
//#include"sensors.h"
#include "motor_driver.h"
#include "encoder_driver.h"
#include "diff_controller.h"
#include<ESP32Servo.h>
#define PID_RATE 30
const int PID_INTERVAL = 1000/PID_RATE;
#define AUTO_STOP_INTERVAL 2000
long last_mtr_cmnd = AUTO_STOP_INTERVAL;
#define BAUDRATE 115200
#define AUTO_STOP_INTERVAL 2000
 
#define RIGHT_MOTOR_ENABLE 4
// #define RIGHT_ENC_PIN_A 5
// #define RIGHT_ENC_PIN_B 18
 
#define LEFT_MOTOR_ENABLE 2
// #define LEFT_ENC_PIN_A 15 instd of dis, we use a difrnt convntn which is LEFT_MOTOR_FORWARD and LEFT_MOTOR_BACKARD
// #define LEFT_ENC_PIN_B 25
 
#define LEFT_MOTOR_FORWARD 15
#define LEFT_MOTOR_BACKWARD 25
 
#define RIGHT_MOTOR_FORWARD 5
#define RIGHT_MOTOR_BACKWARD 18
 
volatile long left_enc_pos = 0L;
volatile long right_enc_pos = 0L;
static const int8_t ENC_STATES [] = {0,1,-1,0,-1,0,0,1,1,0,0,-1,0,-1,1,0};  //encoder lookup table
  
Servo my_servo_one;  // create servo object to control a servo
// 16 servo objects can be created on the ESP32
Servo my_servo_two;

Servo my_servo_three;

Servo my_servo_four;

Servo my_servo_five;

//bool considr_rbt_arm_numbr = 0;

int waist_angle = 0;
int sholdr_angle = 0;
int elbow_angle = 0;
int wrst_angle = 0;
int gripr_angle = 0;

bool waist_reachd = 0;
bool sholdr_reachd = 0;
bool elbow_reachd = 0;
bool wrst_reachd = 0;
bool gripr_reachd = 0;

  int servo_pin_one = 12;
int servo_pin_two = 13;
int servo_pin_three = 14;
int servo_pin_four = 26;
int servo_pin_five = 27;

int waist_angle_old = 0;
int sholdr_angle_old = 0;
int elbow_angle_old = 0;
int wrst_angle_old = 0;
int gripr_angle_old = 0;

// Character arrays to hold the first and second arguments
char argv1[16];
char argv2[16];



void move_rbt_arm()
{
    //considr_rbt_arm_numbr = 1;
    std::string st(argv1);

      waist_angle = atoi(st.substr(0,3).c_str());
      sholdr_angle = atoi(st.substr(3,3).c_str());
      elbow_angle = atoi(st.substr(6,3).c_str());
      wrst_angle = atoi(st.substr(9,3).c_str());
      gripr_angle = atoi(st.substr(12,3).c_str());

  Serial.print(" waist is ");
  Serial.print(waist_angle);
  Serial.print(" shldr is ");
  Serial.print(sholdr_angle);
  Serial.print(" elbw is ");
  Serial.print(elbow_angle);
  Serial.print(" wrst is ");
  Serial.print(wrst_angle);
  Serial.print(" grpr is ");
  Serial.println(gripr_angle);

      if(waist_angle_old < waist_angle)
      {
        for (int pos = waist_angle_old; pos <= waist_angle; pos += 1)
        { // goes from 0 degrees to 180 degrees
          // in steps of 1 degree
          my_servo_one.write(pos);    // tell servo to go to position in variable 'pos'
          delay(15);             // waits 15ms for the servo to reach the position
        }
      }
      if(waist_angle_old > waist_angle)
      {
        for (int pos = waist_angle_old; pos >= waist_angle; pos -= 1)
        { // goes from 180 degrees to 0 degrees
          my_servo_one.write(pos);    // tell servo to go to position in variable 'pos'
          delay(15);             // waits 15ms for the servo to reach the position
        }
      }

      if(sholdr_angle_old < sholdr_angle)
      {
          for (int pos = sholdr_angle_old; pos <= sholdr_angle; pos += 1)
          { // goes from 0 degrees to 180 degrees
            // in steps of 1 degree
            my_servo_two.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }
      if(sholdr_angle_old > sholdr_angle)
      {
          for (int pos = sholdr_angle_old; pos >= sholdr_angle; pos -= 1)
          { // goes from 180 degrees to 0 degrees
            my_servo_two.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }

      if(elbow_angle_old < elbow_angle)
      {
          for (int pos = elbow_angle_old; pos <= elbow_angle; pos += 1)
          { // goes from 0 degrees to 180 degrees
            // in steps of 1 degree
            my_servo_three.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }
      if(elbow_angle_old > elbow_angle)
      {
          for (int pos = elbow_angle_old; pos >= elbow_angle; pos -= 1)
          { // goes from 180 degrees to 0 degrees
            my_servo_three.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }

      if(wrst_angle_old < wrst_angle)
      {
          for (int pos = wrst_angle_old; pos <= wrst_angle; pos += 1)
          { // goes from 0 degrees to 180 degrees
            // in steps of 1 degree
            my_servo_four.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }
      if(wrst_angle_old > wrst_angle)
      {
          for (int pos = wrst_angle_old; pos >= wrst_angle; pos -= 1)
          { // goes from 180 degrees to 0 degrees
            my_servo_four.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }

      if(gripr_angle_old < gripr_angle)
      {
          for (int pos = gripr_angle_old; pos <= gripr_angle; pos += 1)
          { // goes from 0 degrees to 180 degrees
            // in steps of 1 degree
            my_servo_five.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }
      if(gripr_angle_old > gripr_angle)
      {
          for (int pos = gripr_angle_old; pos >= gripr_angle; pos -= 1)
          { // goes from 180 degrees to 0 degrees
            my_servo_five.write(pos);    // tell servo to go to position in variable 'pos'
            delay(15);             // waits 15ms for the servo to reach the position
          }
      }

      waist_angle_old = waist_angle;
      sholdr_angle_old = sholdr_angle;
      elbow_angle_old = elbow_angle;
      wrst_angle_old = wrst_angle;
      gripr_angle_old = gripr_angle;
  

  // if(waist_reachd == 1 && sholdr_reachd == 1 && elbow_reachd == 1 && wrst_reachd == 1 && gripr_reachd == 1)
  // { dis step isnt necesary now since we arnt using considr_rbt_arm_numbr
  //   considr_rbt_arm_numbr = 0;
  // }

}


//stuff for L298_MOTOR_DRIVER
void initMotorController()
{
    digitalWrite(RIGHT_MOTOR_ENABLE, HIGH); // I think dis is an error since initMotorController is called in void_setup without any restrictions! MBS
    digitalWrite(LEFT_MOTOR_ENABLE, HIGH);
    // pinMode(LEFT_MOTOR_PIN_DIR,OUTPUT);
    // pinMode(RIGHT_MOTOR_PIN_DIR,OUTPUT);
}
void setMotorSpeed(int i, int spd)
{
    unsigned char reverse = 0;
    if (spd < 0)
    {
      spd = -spd;
      reverse = 1;
    }
    if (spd > 255)
    {
      spd = 255;
    }
    if(i ==LEFT)
    {
 
      if(reverse == 0)
      {
        analogWrite(LEFT_MOTOR_FORWARD, spd); //LEFT_MOTOR_FORWARD is LEFT_MOTOR_FORWARD_PIN_NUMBER
        analogWrite(LEFT_MOTOR_BACKWARD, 0);  //LEFT_MOTOR_BACKWARD is LEFT_MOTOR_BACKWARD_PIN_NUMBER just wrtn hre for cnvence
      }
      else if(reverse == 1)
      {
        analogWrite(LEFT_MOTOR_BACKWARD, spd);
        analogWrite(LEFT_MOTOR_FORWARD, 0);
      }   
    }
    else
    {
      if(reverse == 0)
      {
        analogWrite(RIGHT_MOTOR_FORWARD, spd);
        analogWrite(RIGHT_MOTOR_BACKWARD, 0);
      }
      else if (reverse == 1)
      {
        analogWrite(RIGHT_MOTOR_BACKWARD, spd);
        analogWrite(RIGHT_MOTOR_FORWARD, 0);
      }    
    }
}
void setMotorSpeeds(int leftSpeed, int rightSpeed)
{
    Serial.print("dis part is exctd");
    setMotorSpeed(LEFT,leftSpeed);
    setMotorSpeed(RIGHT,rightSpeed);
}
 
 
#define USE_BASE      // Enable the base controller code
//#undef USE_BASE     // Disable the base controller code
 
/* Define the motor controller and encoder library you are using */
#ifdef USE_BASE
   /* The Pololu VNH5019 dual motor driver shield */
   //#define POLOLU_VNH5019
 
   /* The Pololu MC33926 dual motor driver shield */
   //#define POLOLU_MC33926
 
   /* The RoboGaia encoder shield */
   //#define ROBOGAIA
   
   /* Encoders directly attached to Arduino board */
   #define ARDUINO_ENC_COUNTER
 
   /* L298 Motor driver*/
   #define L298_MOTOR_DRIVER
#endif
 
//#define USE_SERVOS  // Enable use of PWM servos as defined in servos.h
#undef USE_SERVOS     // Disable use of PWM servos
 
/* Serial port baud rate */
#define BAUDRATE     115200
 
/* Maximum PWM signal */
#define MAX_PWM        255
 
#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif
 
/* Include definition of serial commands */
 
 
/* Sensor functions */
//#include "sensors.h"
 
/* Include servo support if required */
#ifdef USE_SERVOS
   #include <Servo.h>
   #include "servos.h"
#endif
 
#ifdef USE_BASE
  /* Motor driver function definitions */
 
  /* Run the PID loop at 30 times per second */
  #define PID_RATE           30     // Hz
 
  /* Convert the rate into an interval */
  // const int PID_INTERVAL = 1000 / PID_RATE;
  
  /* Track the next time we make a PID calculation */
  unsigned long nextPID = PID_INTERVAL;
 
  /* Stop the robot if it hasn't received a movement command
   in this number of milliseconds */
  #define AUTO_STOP_INTERVAL 2000
  long lastMotorCommand = AUTO_STOP_INTERVAL;
#endif
 
/* Variable initialization */
 
// A pair of varibles to help parse serial commands (thanks Fergs)
int arg = 0;
int indx = 0;
 
// Variable to hold an input character
char chr;
 
// Variable to hold the current single-character command
char cmd;
 
long temp_argv1 = 0;
long temp_argv2 = 0;
 
// The arguments converted to integers
long arg1;
long arg2;
 
/* Clear the current command parameters */
void resetCommand() {
  cmd = NULL;
  memset(argv1, 0, sizeof(argv1));
  memset(argv2, 0, sizeof(argv2));
  arg1 = 0;
  arg2 = 0;
  arg = 0;
  indx = 0;
}
 
/* Run a command.  Commands are defined in commands.h */
void runCommand() {
  Serial.print("inside run command");
  int i = 0;
  char *p = argv1;
  char *str;
  int pid_args[4];
  arg1 = atoi(argv1);
  arg2 = atoi(argv2);
  
  switch(cmd) {
    Serial.println("case begins");
  // case DIGITAL_READ:
  //   Serial.println(digitalRead(arg1));
  //   break;
  case MOTOR_RAW_PWM:
    /* Reset the auto stop timer */
    //lastMotorCommand = millis();
    resetPID();
    Serial.println("inside case");
    moving = 0; // Sneaky way to temporarily disable the PID
    temp_argv1 = arg1;
    temp_argv2 = arg2;
    setMotorSpeeds(arg1, arg2);
    Serial.println("HOWDYY OK");
      break;
  
  case MOVE_ROBOT_ARM:
     move_rbt_arm();
     Serial.println("moving robot arm");
      break;
  default:
    Serial.println("Invalid Command");
    break;
  }
}
 
/* Setup function--runs once at startup. */
void setup() 
{
  Serial.begin(BAUDRATE);
 

   // Allow allocation of all timers
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  my_servo_one.setPeriodHertz(50);    // standard 50 hz servo
  my_servo_one.attach(servo_pin_one, 1000, 2000); // attaches the servo on pin 18 to the servo object
 
  my_servo_two.setPeriodHertz(50);
  my_servo_two.attach(servo_pin_two, 1000,2000);

  my_servo_three.setPeriodHertz(50);
  my_servo_three.attach(servo_pin_three,1000,2000);

  my_servo_four.setPeriodHertz(50);
  my_servo_four.attach(servo_pin_four,500,2400);

  my_servo_five.setPeriodHertz(50);
  my_servo_five.attach(servo_pin_five,500,2400);
// Initialize the motor controller if used */
#ifdef USE_BASE
  // #ifdef ARDUINO_ENC_COUNTER
  //   //set as inputs
  //   DDRD &= ~(1<<LEFT_ENC_PIN_A);
  //   DDRD &= ~(1<<LEFT_ENC_PIN_B);
  //   DDRC &= ~(1<<RIGHT_ENC_PIN_A);
  //   DDRC &= ~(1<<RIGHT_ENC_PIN_B);
    
  //   //enable pull up resistors
  //   PORTD |= (1<<LEFT_ENC_PIN_A);
  //   PORTD |= (1<<LEFT_ENC_PIN_B);
  //   PORTC |= (1<<RIGHT_ENC_PIN_A);
  //   PORTC |= (1<<RIGHT_ENC_PIN_B);
    
  //   // tell pin change mask to listen to left encoder pins
  //   PCMSK2 |= (1 << LEFT_ENC_PIN_A)|(1 << LEFT_ENC_PIN_B);
  //   // tell pin change mask to listen to right encoder pins
  //   PCMSK1 |= (1 << RIGHT_ENC_PIN_A)|(1 << RIGHT_ENC_PIN_B);
    
  //   // enable PCINT1 and PCINT2 interrupt in the general interrupt mask
  //   PCICR |= (1 << PCIE1) | (1 << PCIE2);
  // #endif
  initMotorController(); //MBS, error here!
  resetPID();
#endif
 
/* Attach servos if used */
  #ifdef USE_SERVOS
    int i;
    for (i = 0; i < N_SERVOS; i++) {
      servos[i].initServo(
          servoPins[i],
          stepDelay[i],
          servoInitPosition[i]);
    }
  #endif
}
 
/* Enter the main loop.  Read and parse input from the serial port
   and run any valid commands. Run a PID calculation at the target
   interval and check for auto-stop conditions.
*/
void loop() 
{
  while (Serial.available() > 0) 
  {
    Serial.println("dis is exctd two");
    
    // Read the next character
    chr = Serial.read();
 
    // Terminate a command with a CR
    if (chr == 'v') 
    {
      if (arg == 1) argv1[indx] = NULL;
      else if (arg == 2) argv2[indx] = NULL;
      runCommand();
      resetCommand();
    }
    // Use spaces to delimit parts of the command
    else if (chr == ' ') 
    {
      // Step through the arguments
      if (arg == 0) arg = 1;
      else if (arg == 1)  
      {
        argv1[indx] = NULL;
        arg = 2;
        indx = 0;
      }
      continue;
    }
    else 
    {
      if (arg == 0) 
      {
        // The first arg is the single-letter command
        cmd = chr;
      }
      else if (arg == 1) 
      {
        // Subsequent arguments can be more than one character
        argv1[indx] = chr;
        indx++;
      }
      else if (arg == 2) 
      {
        argv2[indx] = chr;
        indx++;
      }
    }
  //   delay(400);
  // Serial.print("argv1 is ");
  // Serial.print(argv1);
  // Serial.print("argv2 is ");
  // Serial.println(argv2);
}
  // delay(200);
  // Serial.print(" temp argv1 is ");
  // Serial.print(temp_argv1);
  // Serial.print(" temp argv2 is ");
  // Serial.print(temp_argv2);
  // Serial.print(" waist is ");
  // Serial.print(waist_angle);
  // Serial.print(" shldr is ");
  // Serial.print(sholdr_angle);
  // Serial.print(" elbw is ");
  // Serial.print(elbow_angle);
  // Serial.print(" wrst is ");
  // Serial.print(wrst_angle);
  // Serial.print(" grpr is ");
  // Serial.println(gripr_angle);
// If we are using base control, run a PID calculation at the appropriate intervals
#ifdef USE_BASE
  // if (millis() > nextPID) {
  //   updatePID();
  //   nextPID += PID_INTERVAL;
  // }
  
  // // Check to see if we have exceeded the auto-stop interval
  // if ((millis() - lastMotorCommand) > AUTO_STOP_INTERVAL) {
  //   setMotorSpeeds(0, 0);
  //   moving = 0;
  // }
#endif
 
// Sweep servos
#ifdef USE_SERVOS
  int i;
  for (i = 0; i < N_SERVOS; i++) {
    servos[i].doSweep();
  }
#endif
}
 