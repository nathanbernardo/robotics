#include <Servo.h> 
#include <AccelStepper.h>

// base motor
#define X_STEP_PIN         54
#define X_DIR_PIN          55
#define X_ENABLE_PIN       38
#define X_MIN_PIN           3
#define X_MAX_PIN           2

// pitch motor
#define Z_STEP_PIN         46
#define Z_DIR_PIN          48
#define Z_ENABLE_PIN       62
#define Z_MIN_PIN          18
#define Z_MAX_PIN          19

// yaw motor
#define E0_STEP_PIN        26
#define E0_DIR_PIN         28
#define E0_ENABLE_PIN      24

// roll motor
#define E1_STEP_PIN        36
#define E1_DIR_PIN         34
#define E1_ENABLE_PIN      30

// shoulder motor
#define DM556T_PULSE       16
#define DM556T_DIR         17

// third motor
#define DM542_DIR          23
#define DM542_PULSE        25
#define SERVO_DRIVER_SCC   27
#define SERVO_DRIVER_SDA   29

#define LEFT_SERVO_PIN 2
#define RIGHT_SERVO_PIN 15
#define DEFAULT_SPEED 300

void servoMoveHome(Servo& leftServo, Servo& rightServo)
{
    leftServo.write(20);
    rightServo.write(90);
}

void servoGrab(Servo& leftServo, Servo& rightServo)
{
    leftServo.write(60);
    rightServo.write(50);
}

void relativeMove(AccelStepper& stepper, float steps)
{
  stepper.move(steps);
  while(abs(stepper.distanceToGo()) > 0)
  {
    if (steps < 0) stepper.setMaxSpeed(-DEFAULT_SPEED);
    else stepper.setMaxSpeed(DEFAULT_SPEED);
    stepper.run();
  }
}

void absoluteMove(AccelStepper& stepper, float targetPosition)
{
  stepper.moveTo(targetPosition);
  while (stepper.distanceToGo() != 0)
  {
    if (stepper.currentPosition() < targetPosition) stepper.setMaxSpeed(DEFAULT_SPEED);
    else stepper.setMaxSpeed(-DEFAULT_SPEED);
    stepper.run();
  }
}

enum ParsingState {
  LOOKING_FOR_SYNC_BYTE,
  PARSING_LENGTH,
  PARSING_PAYLOAD,
  PARSING_END_BYTE,
};

ParsingState parsingState;
AccelStepper baseStepper(1, X_STEP_PIN, X_DIR_PIN);
AccelStepper yawStepper(1, E0_STEP_PIN, E0_DIR_PIN);
AccelStepper rollStepper(1, E1_STEP_PIN, E1_DIR_PIN);
AccelStepper pitchStepper(1, Z_STEP_PIN, Z_DIR_PIN);
AccelStepper shoulderStepper(1, DM556T_PULSE, DM556T_DIR);
AccelStepper thirdStepper(AccelStepper::DRIVER, DM542_PULSE, DM542_DIR);
Servo leftServo;
Servo rightServo;
AccelStepper* stepperArray[6];
bool relativeMode = false;

void setup() {

  stepperArray[0] = &baseStepper;
  stepperArray[1] = &shoulderStepper;
  stepperArray[2] = &thirdStepper;
  stepperArray[3] = &pitchStepper;
  stepperArray[4] = &yawStepper;
  stepperArray[5] = &rollStepper;

  parsingState = LOOKING_FOR_SYNC_BYTE;
  relativeMode = false;

  pinMode(X_ENABLE_PIN    , OUTPUT);
  pinMode(E0_ENABLE_PIN    , OUTPUT);
  pinMode(E1_ENABLE_PIN    , OUTPUT);
  pinMode(Z_ENABLE_PIN    , OUTPUT);

  Serial.begin(9600);

  baseStepper.setMaxSpeed(200);
  baseStepper.setAcceleration(400);
  baseStepper.setCurrentPosition(0);

  yawStepper.setMaxSpeed(200);
  yawStepper.setAcceleration(200);
  yawStepper.setCurrentPosition(0);

  rollStepper.setMaxSpeed(200);
  rollStepper.setAcceleration(200);
  rollStepper.setCurrentPosition(0);

  shoulderStepper.setMaxSpeed(300);
  shoulderStepper.setAcceleration(300);
  shoulderStepper.setCurrentPosition(0);

  pitchStepper.setMaxSpeed(100);
  pitchStepper.setAcceleration(100);
  pitchStepper.setCurrentPosition(0);

  thirdStepper.setMaxSpeed(100);
  thirdStepper.setAcceleration(100);
  thirdStepper.setCurrentPosition(0);

  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);
}

void loop() {
  int byte = Serial.read();
  delay(50);

  if (byte == 0xAA && parsingState == LOOKING_FOR_SYNC_BYTE) {
    parsingState = PARSING_LENGTH;
  }

  int ACTUATOR_MESSAGE_LENGTH = 27;
  char buffer[2048];
  if (Serial.available() >= ACTUATOR_MESSAGE_LENGTH - 1 && parsingState == PARSING_LENGTH) {
    Serial.readBytes(buffer, Serial.available());
    int j = 1;
    for (int iter = 0; iter < 6; iter++) {
      float steps = *((float*)(&buffer[j])); 
      Serial.println("Extracted payload: ");
      Serial.println(steps);
      Serial.flush();

      if (relativeMode) relativeMove(*stepperArray[iter], steps);
      else absoluteMove(*stepperArray[iter], steps);

      j = j + sizeof(float);
    }
    parsingState = LOOKING_FOR_SYNC_BYTE;
  }
}