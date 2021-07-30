/*
 * GLUT Shapes Demo
 *
 * Written by Nigel Stewart November 2003
 *
 * This program is test harness for the sphere, cone
 * and torus shapes in GLUT.
 *
 * Spinning wireframe and smooth shaded shapes are
 * displayed until the ESC or q key is pressed.  The
 * number of geometry stacks and slices can be adjusted
 * using the + and - keys.
 */
#include<windows.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdlib.h>




static int shoulder = 0, elbow = 0, fingerBase = 0, fingerUp = 0,fingerBase1 = 0, fingerUp1 = 0,fingerBase2 = 0, fingerUp2 = 0,fingerBase3 = 0, fingerUp3 = 0
,shoulderR = 0, elbowR = 0,chest=0, fingerBaseR = 0, LegBaseRA=0,LegBaseLA=0,fingerUpR = 0,shoulderA=0,fingerBase1R = 0,shoulderRF=0,shoulderzR=0, fingerUp1R = 0,fingerBase2R = 0, fingerUp2R = 0,fingerBase3R = 0, fingerUp3R = 0,LegBaseR = 0,LegDownR=0,LegBaseL = 0,LegDownL=0,FootR=0,FootL=0;
int moving, startx, starty;


GLfloat angle = 0.0;   /* in degrees */
GLfloat angle2 = 0.0;   /* in degrees */


void init(void)
{
   glClearColor(0.0, 0.0, 0.0, 0.0);
   glShadeModel(GL_FLAT);
}

void display(void)
{
   glClear(GL_COLOR_BUFFER_BIT);

glPushMatrix();
   glPushMatrix();
   glRotatef(angle2, 1.0, 0.0, 0.0);
   glRotatef(angle, 0.0, 1.0, 0.0);
   glPushMatrix();
   glTranslatef (-1.5, 0.0, 0.0);
   glRotatef ((GLfloat) shoulder, 0.0, 0.0, 1.0);
   glRotatef ((GLfloat) shoulderA, -1.0, 0.0, -1.0);

   glTranslatef (0.5, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.0, 0.3, 0.5);
   glutWireCube (1);
   glPopMatrix();
   glTranslatef (0.5, 0.0, 0.0);
   glRotatef ((GLfloat) elbow, 0.0, 0.0, 1.0);
   glTranslatef (0.5, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.0, 0.3, 0.5);
   glutWireCube (1);
   glPopMatrix();

//////chest
  // glPushMatrix();
   //glPushMatrix();
   //glTranslatef (-1.0, 0.0, -0.8);
   //glRotatef ((GLfloat) chest, 0.0, 0.0, 1.0);
   //glTranslatef (0.5, 0.0, 0.0);
   //glPushMatrix();
   //glScalef (2.0, 1.0, 1.0);
   //glutWireCube (1);
   //glPopMatrix();
   //glPopMatrix();



/////////rightleg
   //glPushMatrix();
   //glPushMatrix();
   //glTranslatef (1.5, 0.0, -1.1);
   //glRotatef ((GLfloat) shoulder, 0.0, 0.0, 1.0);
   //glTranslatef (-0.9, 0.0, 0.0);
   //glPushMatrix();
   //glScalef (2.0, 0.5, 0.5);
   //glutWireCube (1.0);
   //glPopMatrix();
   //glTranslatef (-0.9, 0.0, 0.0);
   //glRotatef ((GLfloat) elbow, 0.0, 0.0, 1.0);
   //glTranslatef (-0.9, 0.0, 0.0);
   //glPushMatrix();
   //glScalef (2.0, 0.5, 0.5);
   //glutWireCube (1.0);
   //glPopMatrix();



//Draw finger flang 1
   glPushMatrix();
   glTranslatef(0.4, 0.0, 0.0);
   glRotatef((GLfloat)fingerBase, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 1
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();

   //Draw finger flang 2
   glPushMatrix();
   glTranslatef(0.4, 0.15, 0.0);
   glRotatef((GLfloat)fingerBase1, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 2
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp1, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();


//Draw finger flang 3
    glPushMatrix();
   glTranslatef(0.4, -0.17, 0.0);
   glRotatef((GLfloat)fingerBase2, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 3
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp2, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();

   //Draw finger flang 4
   glPushMatrix();
   glTranslatef(0.4, 0.15, -0.2);
   glRotatef((GLfloat)fingerBase3, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 4
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp3, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();




   glPopMatrix();







   //glPushMatrix();

   //glTranslatef (-0.5, 0.0, 0.0);
   //glRotatef ((GLfloat) elbow, 0.0, 0.0, 1.0);
   //glTranslatef (0.5, 0.0, 0.0);
   //glPushMatrix();
   //glScalef (2.0, 0.6, 1.0);
   //glutWireCube (0.5);
   //glPopMatrix();
   //glPopMatrix();
   glPushMatrix();
   glTranslatef (-1.5, 0.0, -1.5);
   glRotatef ((GLfloat) shoulderRF, 0.0, 0.0, 1.0);
   glRotatef ((GLfloat) shoulderR, -1.0, 0.0, 1.0);
   glTranslatef (0.5, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.0, 0.3, 0.5);
   glutWireCube (1);
   glPopMatrix();
   glTranslatef (0.5, 0.0, 0.0);
   glRotatef ((GLfloat) elbowR, 0.0, 0.0, 1.0);
   glTranslatef (0.5, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.0, 0.3, 0.5);
   glutWireCube (1);
   glPopMatrix();



//Draw finger flang 1
   glPushMatrix();
   glTranslatef(0.4, 0.0, -0.2);
   glRotatef((GLfloat)fingerBaseR, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 1
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUpR, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();

   //Draw finger flang 2
   glPushMatrix();
   glTranslatef(0.4, 0.15, -0.2);
   glRotatef((GLfloat)fingerBase1R, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 2
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp1R, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();


//Draw finger flang 3
    glPushMatrix();
   glTranslatef(0.4, -0.17, -0.2);
   glRotatef((GLfloat)fingerBase2R, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 3
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp2R, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();

   //Draw finger flang 4
   glPushMatrix();
   glTranslatef(0.4, 0.15, 0.0);
   glRotatef((GLfloat)fingerBase3R, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();


   //Draw finger flang 4
   glTranslatef(0.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp3R, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(0.5);
   glPopMatrix();
   glPopMatrix();
glPopMatrix();

////////// the face shape




   glPushMatrix();


   glPushMatrix();
   glPushMatrix();
   glTranslatef (-1.0, 0.0, -0.7);
   glRotatef ((GLfloat) chest, 0.0, 0.0, 1.0);
   glTranslatef (0.5, 0.0, 0.0);
   glPushMatrix();
   glScalef (2.0, 1.0, 1.0);
   glutWireCube (1);
   glPopMatrix();
   glPopMatrix();
   glPopMatrix();


   glPushMatrix();


   glPushMatrix();


        glTranslated(-2.5, 0.0, -0.7);
        glutWireSphere(0.5,9,9);
        glPopMatrix();
   glPushMatrix();
   glTranslatef (0.45,0.0, -0.55);
   glRotatef ((GLfloat) LegBaseL, 0.0, 0.0, 1.0);
   glRotatef ((GLfloat) LegBaseLA, -1.0, 0.0, -1.0);
   glTranslatef (0.759, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.5, 0.65, 0.55);
   glutWireCube (1.0);
   glPopMatrix();
   //////the lower part of left leg
 //  glPushMatrix();
   glTranslatef (0.759, 0.0, 0.0);
   glRotatef ((GLfloat) LegDownL, 0.0, 0.0, 1.0);
   glTranslatef (0.759, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.5, 0.65, 0.55);
   glutWireCube (1.0);
   glPopMatrix();
   //////////left foot
   glPushMatrix();
   glTranslatef (0.6, 0.0, 0.0);

   glRotatef ((GLfloat) FootL, 0.0, 0.0, 1.0);
   glTranslatef (0.5, 0.0, 0.0);
    glPushMatrix();
   glScalef (0.7, 1.15, 0.75);
   glutWireCube (1.0);

glPopMatrix();
   glPopMatrix();
      glPopMatrix();




   glPushMatrix();

   glTranslatef (0.45,0.0, -1.0);
   glRotatef ((GLfloat) LegBaseR, 0.0, 0.0, 1.0);
   glRotatef ((GLfloat) LegBaseRA, -1.0, 0.0, 1.0);

   glTranslatef (0.759, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.5, 0.65, 0.55);
   glutWireCube (1.0);
   glPopMatrix();
   //////the lower part of left leg
   // glPushMatrix();
   glTranslatef (0.759, 0.0, 0.0);
   glRotatef ((GLfloat) LegDownR, 0.0, 0.0, 1.0);

   glTranslatef (0.759, 0.0, 0.0);
   glPushMatrix();
   glScalef (1.5, 0.65, 0.55);
   glutWireCube (1.0);
   glPopMatrix();
   //////////left foot
   glPushMatrix();
   glTranslatef (0.6, 0.0, 0.0);

   glRotatef ((GLfloat) FootR, 0.0, 0.0, 1.0);
   glTranslatef (0.5, 0.0, 0.0);
    glPushMatrix();
   glScalef (0.7, 1.15, 0.75);
   glutWireCube (1.0);
   glPopMatrix();
   //glTranslated(-2.5, 0.0, -0.7);
     //   glutWireSphere(0.5,9,9);


   glPopMatrix();


glPopMatrix();


//glPushMatrix();


        //glTranslated(-2.5, 0.0, -0.7);
        //glutWireSphere(0.5,9,9);
           // glPopMatrix();

    glPopMatrix();



glPopMatrix();







////////// the upper body
  // glPushMatrix();
   //glTranslatef (-1.2, 0.0, -0.76);

   //glRotatef ((GLfloat) elbow, 0.0, 0.0, 1.0);
  // glTranslatef (0.5, 0.0, 0.0);
   //glScalef (2.0, 0.6, 1.0);
   //glutWireCube (1.0);
   //glPopMatrix();
///////////the lower body









///////////////left fingers
   //Draw finger flang 1





//////////// right fingers

glPopMatrix();


   glutSwapBuffers();
}

void reshape(int w, int h)
{
   glViewport(0, 0, (GLsizei)w, (GLsizei)h);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(85.0, (GLfloat)w / (GLfloat)h, 1.0, 20.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glTranslatef(0.0, 0.0, -5.0);
}

void keyboard(unsigned char key, int x, int y)
{
   switch (key)
   {




case '1':
      shoulderA= (shoulderA + 5) % 360;
      glutPostRedisplay();
      break;
   case '2':
      shoulderA= (shoulderA - 5) % 360;
      glutPostRedisplay();
      break;
case '3':
      LegBaseRA = (LegBaseRA + 5) % 360;
      glutPostRedisplay();
      break;
   case '4':
      LegBaseRA = (LegBaseRA - 5) % 360;
      glutPostRedisplay();
      break;
case '5':
      LegBaseLA = (LegBaseLA + 5) % 360;
      glutPostRedisplay();
      break;
   case '6':
      LegBaseLA = (LegBaseLA - 5) % 360;
      glutPostRedisplay();
      break;
 case '7':
      shoulderRF = (shoulderRF + 5) % 360;
      glutPostRedisplay();
      break;
   case '8':
      shoulderRF = (shoulderRF - 5) % 360;
      glutPostRedisplay();
      break;
   case 's':
      shoulder = (shoulder + 5) % 360;
      glutPostRedisplay();
      break;
   case 'S':
      shoulder = (shoulder - 5) % 360;
      glutPostRedisplay();
      break;
    case 'a':
      shoulderR = (shoulderR + 5) % 360;
      glutPostRedisplay();
      break;
   case 'A':
      shoulderR = (shoulderR - 5) % 360;
      glutPostRedisplay();
      break;
   case 'D':
      elbow = (elbow + 5) % 180;
      glutPostRedisplay();
      break;
   case 'B':
      elbow = (elbow - 5) % 10;
      glutPostRedisplay();
      break;
    case 'c':
      elbowR = (elbowR + 5) % 180;
      glutPostRedisplay();
      break;
   case 'C':
      elbowR = (elbowR - 5) % 10;
      glutPostRedisplay();
      break;

case 'e':
      LegBaseL = (LegBaseL + 5) % 90;
      glutPostRedisplay();
      break;
   case 'E':
      LegBaseL = (LegBaseL - 5) % 60;
      glutPostRedisplay();
      break;
   case 'd':
      LegDownL = (LegDownL + 5) % 10;
      glutPostRedisplay();
      break;
   case 'F':
      LegDownL = (LegDownL - 5) % 150;
      glutPostRedisplay();
      break;
   case 'g':
      FootL = (FootL + 5) % 30;
      break;
   case 'G':
      FootL = (FootL - 5) %10;
      glutPostRedisplay();
      break;

   case 'h':
      LegBaseR = (LegBaseR + 5) % 90;
      glutPostRedisplay();
      break;
   case 'H':
      LegBaseR = (LegBaseR - 5) % 60;
      glutPostRedisplay();
      break;
   case 'i':
      LegDownR = (LegDownR + 5) % 10;
      glutPostRedisplay();
      break;
   case 'I':
      LegDownR = (LegDownR - 5) % 150;
      glutPostRedisplay();
      break;
   case 'j':
      FootR = (FootR + 5) % 30;
      glutPostRedisplay();
      break;
   case 'J':
      FootR = (FootR- 5) %10;
      glutPostRedisplay();
      break;
   case 'k':
      fingerBase = (fingerBase + 5) % 30;
      glutPostRedisplay();
      break;
   case 'K':
      fingerBase = (fingerBase - 5) % 30;
      glutPostRedisplay();
      break;
   case 'l':
      fingerUp = (fingerUp + 5) % 30;
      glutPostRedisplay();
      break;
   case 'L':
      fingerUp = (fingerUp - 5) % 30;
      glutPostRedisplay();
      break;
   case 'm':
      fingerBase1 = (fingerBase1 + 5) % 90;
      glutPostRedisplay();
      break;
   case 'M':
      fingerBase1 = (fingerBase1 - 5) %90;
      glutPostRedisplay();
      break;
   case 'n':
      fingerUp1 = (fingerUp1 + 5) % 90;
      glutPostRedisplay();
      break;
   case 'N':
      fingerUp1 = (fingerUp1 - 5) % 90;
      glutPostRedisplay();
      break;
   case 'o':
      fingerBase2 = (fingerBase2 + 5) % 90;
      glutPostRedisplay();
      break;
   case 'O':
      fingerBase2 = (fingerBase2 - 5) % 90;
      glutPostRedisplay();
      break;
   case 'p':
      fingerUp2 = (fingerUp2 + 5) % 90;
      glutPostRedisplay();
      break;
   case 'P':
      fingerUp2 = (fingerUp2 - 5) % 90;
      glutPostRedisplay();
      break;
   case 'q':
      fingerBase3 = (fingerBase3 + 5) % 90;
      glutPostRedisplay();
      break;
   case 'Q':
      fingerBase3 = (fingerBase3 - 5) % 90;
      glutPostRedisplay();
      break;
   case 'r':
      fingerUp3 = (fingerUp3 + 5) % 90;
      glutPostRedisplay();
      break;
   case 'R':
      fingerUp3 = (fingerUp3 - 5) % 90;
      glutPostRedisplay();
      break;


	default:
		break;
   }
}

static void mouse(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_DOWN) {
      moving = 1;
      startx = x;
      starty = y;
    }
    if (state == GLUT_UP) {
      moving = 0;
    }
  }
}


static void motion(int x, int y)
{
  if (moving) {
    angle = angle + (x - startx);
    angle2 = angle2 + (y - starty);
    startx = x;
    starty = y;
    glutPostRedisplay();
  }
}



int main(int argc, char **argv)
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
   glutInitWindowSize(500, 500);
   glutInitWindowPosition(100, 100);
   glutCreateWindow(argv[0]);
   init();
   glutMouseFunc(mouse);
   glutMotionFunc(motion);
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutKeyboardFunc(keyboard);
   glutMainLoop();
   return 0;
}
