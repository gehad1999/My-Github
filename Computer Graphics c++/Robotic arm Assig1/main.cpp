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


static int shoulder = 0, elbow = 0, fingerBase = 0, fingerUp = 0,fingerBase1 = 0, fingerUp1 = 0,fingerBase2 = 0, fingerUp2 = 0,fingerBase3 = 0, fingerUp3 = 0;
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
   glRotatef(angle2, 1.0, 0.0, 0.0);
   glRotatef(angle, 0.0, 1.0, 0.0);
   glTranslatef (-1.0, 0.0, 0.0);
   glRotatef ((GLfloat) shoulder, 0.0, 0.0, 1.0);
   glTranslatef (1.0, 0.0, 0.0);
   glPushMatrix();
   glScalef (2.0, 0.6, 1.0);
   glutWireCube (1.0);
   glPopMatrix();
   glTranslatef (1.0, 0.0, 0.0);
   glRotatef ((GLfloat) elbow, 0.0, 0.0, 1.0);
   glTranslatef (1.0, 0.0, 0.0);
   glPushMatrix();
   glScalef (2.0, 0.6, 1.0);
   glutWireCube (1.0);
   glPopMatrix();

   //Draw finger flang 1
   glPushMatrix();
   glTranslatef(1.0, 0.0, 0.0);
   glRotatef((GLfloat)fingerBase, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();


   //Draw finger flang 1
   glTranslatef(0.15, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();
   glPopMatrix();

   //Draw finger flang 2
   glPushMatrix();
   glTranslatef(1.0, 0.15, 0.0);
   glRotatef((GLfloat)fingerBase1, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();


   //Draw finger flang 2
   glTranslatef(0.15, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp1, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();
   glPopMatrix();


//Draw finger flang 3
    glPushMatrix();
   glTranslatef(1.0, -0.17, 0.0);
   glRotatef((GLfloat)fingerBase2, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();


   //Draw finger flang 3
   glTranslatef(0.15, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp2, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();
   glPopMatrix();

   //Draw finger flang 4
   glPushMatrix();
   glTranslatef(1.0, 0.15, -0.2);
   glRotatef((GLfloat)fingerBase3, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();


   //Draw finger flang 4
   glTranslatef(0.15, 0.0, 0.0);
   glRotatef((GLfloat)fingerUp3, 0.0, 0.0, 1.0);
   glTranslatef(0.15, 0.0, 0.0);
   glPushMatrix();
   glScalef(0.3, 0.1, 0.1);
   glutWireCube(1);
   glPopMatrix();
   glPopMatrix();

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
   case 's':
      shoulder = (shoulder + 5) % 360;
      glutPostRedisplay();
      break;
   case 'S':
      shoulder = (shoulder - 5) % 360;
      glutPostRedisplay();
      break;
   case 'e':
      elbow = (elbow + 5) % 180;
      glutPostRedisplay();
      break;
   case 'E':
      elbow = (elbow - 5) % 10;
      glutPostRedisplay();
      break;
   case 'a':
      fingerBase = (fingerBase + 5) % 30;
      glutPostRedisplay();
      break;
   case 'A':
      fingerBase = (fingerBase - 5) % 30;
      glutPostRedisplay();
      break;
   case 'b':
      fingerUp = (fingerUp + 5) % 30;
      //fingerUp1 = (fingerUp1 + 5) % 10;
      //fingerUp2 = (fingerUp2 + 5) % 10;

      glutPostRedisplay();
      break;
   case 'B':
      fingerUp = (fingerUp - 5) % 30;
      //fingerUp1 = (fingerUp1 - 5) % 20;
      //fingerUp2 = (fingerUp2 - 5) % 20;
      glutPostRedisplay();
      break;
   case 'c':
      fingerBase1 = (fingerBase1 + 5) % 30;
      //fingerBase2 = (fingerBase2 + 5) % 20;
      //fingerBase = (fingerBase + 8) % 32;
      glutPostRedisplay();
      break;
   case 'C':
      fingerBase1 = (fingerBase1 - 5) %20;
      //fingerBase2 = (fingerBase2 - 5) %20;
      //fingerBase = (fingerBase - 8) %32;
      glutPostRedisplay();
      break;
   case 'd':
      fingerUp1 = (fingerUp1 + 5) % 30;
      //fingerUp = (fingerUp + 5) % 30;
      //fingerUp2 = (fingerUp2 + 5) % 30;

      glutPostRedisplay();
      break;
   case 'D':
      fingerUp1 = (fingerUp1 - 5) % 30;
      //fingerUp = (fingerUp - 5) % 30;
      //fingerUp2 = (fingerUp2 - 5) % 30;
      glutPostRedisplay();
      break;
   case 'l':
      fingerBase2 = (fingerBase2 + 5) % 20;
      glutPostRedisplay();
      break;
   case 'L':
      fingerBase2 = (fingerBase2 - 5) % 20;
      glutPostRedisplay();
      break;
   case 'f':
      fingerUp2 = (fingerUp2 + 5) % 35;
      glutPostRedisplay();
      break;
   case 'F':
      fingerUp2 = (fingerUp2 - 5) % 35;
      glutPostRedisplay();
      break;
   case 'g':
      fingerBase3 = (fingerBase3 + 5) % 20;
      glutPostRedisplay();
      break;
   case 'G':
      fingerBase3 = (fingerBase3 - 5) % 20;
      glutPostRedisplay();
      break;
   case 'k':
      fingerUp3 = (fingerUp3 + 5) % 35;
      glutPostRedisplay();
      break;
   case 'K':
      fingerUp3 = (fingerUp3 - 5) % 35;
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
