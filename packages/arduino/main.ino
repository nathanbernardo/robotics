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

#define Z_STEP_PIN         46
#define Z_DIR_PIN          48
#define Z_ENABLE_PIN       62
#define Z_MIN_PIN          18
#define Z_MAX_PIN          19

#define LIMIT_PIN       51

#define ONE_REVOLUTION     800*4

#include <AccelStepper.h>
#include <ezButton.h>

ezButton limitSwitch(LIMIT_PIN);

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
AccelStepper stepper1(1, X_STEP_PIN, X_DIR_PIN); // (Type of driver: with 2 pins, STEP, DIR)
AccelStepper stepper2(1, Y_STEP_PIN, Y_DIR_PIN); // (Type of driver: with 2 pins, STEP, DIR)
AccelStepper stepper3(1, Z_STEP_PIN, Z_DIR_PIN); // (Type of driver: with 2 pins, STEP, DIR)

void setup() {
  Serial.begin(9600);
  parsingState = LOOKING_FOR_SYNC_BYTE;

  pinMode(X_ENABLE_PIN    , OUTPUT);
  pinMode(Y_ENABLE_PIN    , OUTPUT);
  pinMode(Z_ENABLE_PIN    , OUTPUT);
  // Set maximum speed value for the stepper
  stepper1.setMaxSpeed(500);
  stepper1.setAcceleration(500);
  stepper1.setCurrentPosition(0);

  stepper2.setMaxSpeed(1000);
  stepper2.setAcceleration(1000);
  stepper2.setCurrentPosition(0);

  stepper3.setMaxSpeed(500);
  stepper3.setAcceleration(500);
  stepper3.setCurrentPosition(0);

  limitSwitch.setDebounceTime(50); // set debounce time to 50 milliseconds
}

void loop() {
  limitSwitch.loop();
 
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
        stepper1.move(uh);
        while(abs(stepper1.distanceToGo()) > 0)
        {
          if (uh < 0)
          {
            stepper1.setMaxSpeed(-500);
          }
          else
          {
            stepper1.setMaxSpeed(500);
          }
          stepper1.run();
        }
      }
      if (iter == 1)
      {
        stepper2.move(uh);
        while(abs(stepper2.distanceToGo()) > 0)
        {
          if (uh < 0)
            stepper2.setSpeed(-800);
          else
            stepper2.setSpeed(800);
            stepper2.run();
          
        }
      }

      if (iter == 2)
      {
        stepper3.move(uh);
        while(abs(stepper3.distanceToGo()) > 0)
        {
          if (uh < 0)
            stepper3.setSpeed(-500);
          else
            stepper3.setSpeed(500);
          stepper3.run();
        }
      }

      j = j + sizeof(float);
    }
    parsingState = LOOKING_FOR_SYNC_BYTE;
  }
}
