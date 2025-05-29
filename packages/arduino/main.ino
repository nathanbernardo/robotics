// base motor
#define X_STEP_PIN         54
#define X_DIR_PIN          55
#define X_ENABLE_PIN       38
#define X_MIN_PIN           3
#define X_MAX_PIN           2

#define Y_STEP_PIN         60
#define Y_DIR_PIN          61
#define Y_ENABLE_PIN       56
#define Y_MIN_PIN          14
#define Y_MAX_PIN          15

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

#define LIMIT_PIN       51

#define ONE_REVOLUTION     800*4

#include <AccelStepper.h>

enum ParsingState {
  LOOKING_FOR_SYNC_BYTE,
  PARSING_LENGTH,
  PARSING_PAYLOAD,
  PARSING_END_BYTE,
};

ParsingState parsingState;
float actuatorPosition1;
float actuatorPosition2;
float actuatorPosition3;
float actuatorPosition4;
float actuatorPosition5;
float actuatorPosition6;

// Define the stepper motor and the pins that is connected to
AccelStepper baseStepper(1, X_STEP_PIN, X_DIR_PIN); // (Type of driver: with 2 pins, STEP, DIR)
AccelStepper yawStepper(1, E0_STEP_PIN, E0_DIR_PIN); // (Type of driver: with 2 pins, STEP, DIR)
AccelStepper rollStepper(1, E1_STEP_PIN, E1_DIR_PIN); // (Type of driver: with 2 pins, STEP, DIR)
AccelStepper pitchStepper(1, Z_STEP_PIN, Z_DIR_PIN);
AccelStepper shoulderStepper(1, DM556T_PULSE, DM556T_DIR);
AccelStepper thirdStepper(AccelStepper::DRIVER, DM542_PULSE, DM542_DIR);

void setup() {
  Serial.begin(9600);
  parsingState = LOOKING_FOR_SYNC_BYTE;

  pinMode(X_ENABLE_PIN    , OUTPUT);
  pinMode(E0_ENABLE_PIN    , OUTPUT);
  pinMode(E1_ENABLE_PIN    , OUTPUT);
  pinMode(Z_ENABLE_PIN    , OUTPUT);
  // Set maximum speed value for the stepper
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
  

}

void loop() {

  ////////////////////////////////////////
  ////// LOOK FOR ACTUATOR MESSAGE ///////
  ////////////////////////////////////////

  // look for the header first
  int byte = Serial.read();
  delay(50);
  if (byte == 0xAA && parsingState == LOOKING_FOR_SYNC_BYTE) {
    parsingState = PARSING_LENGTH;
  }
  int ACTUATOR_MESSAGE_LENGTH = 27;
  int LENGTH = ACTUATOR_MESSAGE_LENGTH - 1;
  char buffer[2048];
  if (Serial.available() >= ACTUATOR_MESSAGE_LENGTH-1 && parsingState == PARSING_LENGTH) {
    int out = Serial.readBytes(buffer, Serial.available());
    int length = buffer[0];
    int j = 1;
    for (int iter = 0; iter < 6; iter++) {
      float uh = *((float*)(&buffer[j])); 
      Serial.println("Extracted payload: ");
      Serial.println(uh);
      Serial.flush();
      if (iter == 0)
      {
        baseStepper.move(uh);
        while(abs(baseStepper.distanceToGo()) > 0)
        {
          if (uh < 0)
          {
            baseStepper.setMaxSpeed(-400);
          }
          else
          {
            baseStepper.setMaxSpeed(400);
          }
          baseStepper.run();
        }
      }
      if (iter == 1)
      {
        shoulderStepper.move(uh);
        while(abs(shoulderStepper.distanceToGo()) > 0)
        {
          if (uh < 0)
            shoulderStepper.setSpeed(-300);
          else
            shoulderStepper.setSpeed(300);
          shoulderStepper.run();
          
        }
      }

      if (iter == 2)
      {
        thirdStepper.move(uh);
        while(abs(thirdStepper.distanceToGo()) > 0)
        {
          if (uh < 0)
            thirdStepper.setSpeed(-100);
          else
            thirdStepper.setSpeed(100);
          thirdStepper.run();
        }
      }

      if (iter == 3)
      {
        pitchStepper.move(uh);
        while(abs(pitchStepper.distanceToGo()) > 0)
        {
          if (uh < 0)
            pitchStepper.setSpeed(-100);
          else
            pitchStepper.setSpeed(100);
          pitchStepper.run();
        }
      }

      if (iter == 4)
      {
        yawStepper.move(uh);
        while(abs(yawStepper.distanceToGo()) > 0)
        {
          if (uh < 0)
            yawStepper.setSpeed(-100);
          else
            yawStepper.setSpeed(100);
          yawStepper.run();
        }
      }
      if (iter == 5)
      {
        rollStepper.move(uh);
        while(abs(rollStepper.distanceToGo()) > 0)
        {
          if (uh < 0)
            rollStepper.setSpeed(-100);
          else
            rollStepper.setSpeed(100);
          rollStepper.run();
        }
      }

      j = j + sizeof(float);
    }
    parsingState = LOOKING_FOR_SYNC_BYTE;
  }
}