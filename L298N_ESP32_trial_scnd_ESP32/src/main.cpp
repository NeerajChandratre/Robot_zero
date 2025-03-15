  #include<Arduino.h>
  #include<ESP32Servo.h>
//dis is 1st (white cable) USB ESP32
#define EncoderPinA 13 //was 18 // encoder A and B are Blue and Yellow for Left motor
#define EncoderPinB 14  //was 21 
#define EncoderPinC 27   //encoder C and D are Blue and Yellow for Right motor
#define EncoderPinD 26


void updateEncoder();
void updateEncoder_two();
bool flg = 0;
int rpm_one;
int rpm_two;
int motor_left_to_int;
int motor_left_to_int_old = 0;
int motor_right_to_int_old = 0;
int motor_right_to_int;
volatile long Encodervalue=0;
volatile long Encodervalue_two=0;
bool considr_numbr = 0;
double old_time;
double new_time;
double new_time_two;

//rtc.offset;	// get or modify the current offset

void setup() 
{

  pinMode(EncoderPinA, INPUT);// Encoder left pins and it works well
  pinMode(EncoderPinB, INPUT);
  pinMode(EncoderPinC, INPUT); //Encoder right pins
  pinMode(EncoderPinD, INPUT);


  attachInterrupt(digitalPinToInterrupt(EncoderPinA), updateEncoder, RISING); //Belongs to left motor
  attachInterrupt(digitalPinToInterrupt(EncoderPinC), updateEncoder_two, RISING); //Belongs to right motor
  Serial.begin(115200);
}

void loop() 
{

  delay(5);
//  delay(1000);


  Serial.print('e'); //here e stands for esp32. W send data from esp32 to our computer
  Serial.print(Encodervalue); //use dis for communictn wid computer
  Serial.print(',');
  Serial.println(Encodervalue_two);
//  // delay(5);
}
void updateEncoder() //belongs to left motor
{
  if (digitalRead(EncoderPinA)> digitalRead(EncoderPinB))
  {
    Encodervalue++;
  }
  else
  {
    Encodervalue--;
  }
}

void updateEncoder_two() //belongs to right motor
{
  if (digitalRead(EncoderPinC)> digitalRead(EncoderPinD))
  {
    Encodervalue_two++;
  }
  else
  {
    Encodervalue_two--;
  }
}

