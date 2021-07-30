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

#include <GL/glut.h>
#include <math.h>



static int shoulder = 0, elbow = 0, fingerBase = 0, fingerUp = 0,fingerBase1 = 0, fingerUp1 = 0,fingerBase2 = 0, fingerUp2 = 0,fingerBase3 = 0, fingerUp3 = 0
,shoulderR = 0, elbowR = 0,chest=0, fingerBaseR = 0, LegBaseRA=0,LegBaseLA=0,fingerUpR = 0,shoulderA=0,fingerBase1R = 0,shoulderRF=0,shoulderzR=0, fingerUp1R = 0,fingerBase2R = 0, fingerUp2R = 0,fingerBase3R = 0, fingerUp3R = 0,LegBaseR = 0,LegDownR=0,LegBaseL = 0,LegDownL=0,FootR=0,FootL=0;
int moving, startx, starty;


GLfloat angle = 0.0  ;   /* in degrees */
GLfloat angle2 = 0.0;   /* in degrees */

// angle of rotation for the camera direction
float anggle=0.0 ,speed=0.02;
// actual vector representing the camera's direction
float lx=0.0f,lz=-1.0f,ly=0.0f;
// XZ position of the camera
float x=0.0f,z=5.0f,y=1.0f;


float DRot = 90;
float Zmax, Zmin;
float VRot =0.0;




double direction[] = { 0, 0, 0 };
double eye[] = { 0, 0, -20 };
double center[] = { 0, 0, 1 };
double up[] = { 0, 1, 0 };

void init(void)
{
    glMatrixMode(GL_PROJECTION);
	gluPerspective(65.0, (GLfloat)1024 / (GLfloat)869, 1.0, 60.0);
}

void crossProduct(double a[], double b[], double c[])
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

void normalize(double a[])
{
	double norm;
	norm = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
	norm = sqrt(norm);
	a[0] /= norm;
	a[1] /= norm;
	a[2] /= norm;
}

void rotatePoint(double a[], double theta, double p[])
{

	double temp[3];
	temp[0] = p[0];
	temp[1] = p[1];
	temp[2] = p[2];

	temp[0] = -a[2] * p[1] + a[1] * p[2];
	temp[1] = a[2] * p[0] - a[0] * p[2];
	temp[2] = -a[1] * p[0] + a[0] * p[1];

	temp[0] *= sin(theta);
	temp[1] *= sin(theta);
	temp[2] *= sin(theta);

	temp[0] += (1 - cos(theta))*(a[0] * a[0] * p[0] + a[0] * a[1] * p[1] + a[0] * a[2] * p[2]);
	temp[1] += (1 - cos(theta))*(a[0] * a[1] * p[0] + a[1] * a[1] * p[1] + a[1] * a[2] * p[2]);
	temp[2] += (1 - cos(theta))*(a[0] * a[2] * p[0] + a[1] * a[2] * p[1] + a[2] * a[2] * p[2]);

	temp[0] += cos(theta)*p[0];
	temp[1] += cos(theta)*p[1];
	temp[2] += cos(theta)*p[2];

	p[0] = temp[0];
	p[1] = temp[1];
	p[2] = temp[2];

}
















void Left()
{
	// implement camera rotation arround vertical window screen axis to the left
	// used by mouse and left arrow


	double speed = 0.01;
	//double direction[] = { 0, 1, 0 };
	rotatePoint(up, speed, eye);
}

void Right()
{
	// implement camera rotation arround vertical window screen axis to the right
	// used by mouse and right arrow



	double speed = -0.01;
	//double direction[] = { 0, 1, 0 };
	rotatePoint(up, speed, eye);
}

void Up()
{
	// implement camera rotation arround horizontal window screen axis +ve
	// used by up arrow
	double speed = 0.2;
	double c[]={0,0,0};
	direction[0] = center[0] - eye[0];
	direction[1] = center[1] - eye[1];
	direction[2] = center[2] - eye[2];
	crossProduct(direction,up,c);
	normalize(c);
    rotatePoint(c, speed, eye);
    rotatePoint(c, speed,up);

	//center[1] += speed;

    //rotatePoint(eye, speed, up);

}

void Down()
{
	// implement camera rotation arround horizontal window screen axis
	// used by down arrow

	double speed = -0.07;
	double c[]={0,0,0};
	direction[0] = center[0] - eye[0];
	direction[1] = center[1] - eye[1];
	direction[2] = center[2] - eye[2];
	crossProduct(direction,up,c);
	normalize(c);
    rotatePoint(c, speed, eye);
    rotatePoint(c, speed,up);
}

void moveForward()
{
   direction[0] = center[0] - eye[0];
	direction[1] = center[1] - eye[1];
	direction[2] = center[2] - eye[2];



	eye[0]    += direction[0] * speed;
	eye[1]    += direction[1] * speed;
	eye[2]    += direction[2] * speed;

	center[0] += direction[0] * speed;
	center[1] += direction[1] * speed;
	center[2] += direction[2] * speed;

}

void moveBack()
{
    direction[0] = center[0] - eye[0];
	direction[1] = center[1] - eye[1];
	direction[2] = center[2] - eye[2];



	eye[0]    -= direction[0] * speed;
	eye[1]    -= direction[1] * speed;
	eye[2]    -= direction[2] * speed;

	center[0] -= direction[0] * speed;
	center[1] -= direction[1] * speed;
	center[2] -= direction[2] * speed;




}











/*void changeSize(int w, int h)
{

// Prevent a divide by zero, when window is too short
// (you cant make a window of zero width).
if (h == 0)
h = 1;
float ratio = w * 1.0 / h;

// Use the Projection Matrix
glMatrixMode(GL_PROJECTION);

// Reset Matrix
glLoadIdentity();

// Set the viewport to be the entire window
glViewport(0, 0, w, h);

// Set the correct perspective.
gluPerspective(60.0f, ratio, 0.1f, 100.0f);

// Get Back to the Modelview
glMatrixMode(GL_MODELVIEW);
}*/

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2]);

	// draw head

	// draw trunck
	// call the robotic body draw function here





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







glPopMatrix();


	glutSwapBuffers();
}









void Timer(int x){
	// Refresh and redraw
	glutPostRedisplay();
	glutTimerFunc(50, Timer, 0);
}




void correct()
{
	double speed = 0.001;
	if (eye[0]>0)
	{
		eye[0] -= speed;
		center[0] -= speed;
	}
	else
	{
		eye[0] += speed;
		center[0] += speed;
	}

	if (DRot == 0)
	{
		if ((eye[2] >= -1.2 && eye[2] <= -1) || eye[2]>0)
		{
			eye[2] -= speed;
			center[2] -= speed;
		}
		else
		{
			eye[2] += speed;
			center[2] += speed;
		}
	}
	else
	{
		if (eye[2]>0)
		{
			eye[2] -= speed;
			center[2] -= speed;
		}
		else
		{
			eye[2] += speed;
			center[2] += speed;
		}
	}

}
void SetBound()
{
	if (DRot == 0 || eye[0]> 0.15 || eye[0]< -0.15)
	{
		if (eye[2] >= -1)
		{
			Zmax = 0.7;
			Zmin = -0.8;
		}
		else
		{
			Zmax = -1.2;
			Zmin = -2.4;
		}
	}
	else
	{
		Zmax = 0.7;
		Zmin = -2.4;
	}
}
void Timer1(int x){
    // Refresh and redraw
    VRot += 5;
    if (VRot == 360)
        VRot=0;
    glutPostRedisplay();
    glutTimerFunc(50, Timer1, 0);
}

void DTimer1(int x){

	DRot -= 1;
	if (DRot == 0)
		return;
	glutPostRedisplay();
	glutTimerFunc(30, DTimer1, 0);


}

void DTimer2(int x){
	DRot += 1;
	if (DRot == 90)
		return;
	glutPostRedisplay();
	glutTimerFunc(30, DTimer2, 0);
}

void specialKeys(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_LEFT: Left(); break;
	case GLUT_KEY_RIGHT: Right(); break;
	case GLUT_KEY_UP:
	    //if (center[1] <= 1.5)
	     Up();
	     break;
	case GLUT_KEY_DOWN:

	    //if (center[1] >= -1.5)
	     Down(); break;
	}

	glutPostRedisplay();
}








/*void processSpecialKeys(int key, int xx, int yy) {

	float fraction = 0.1f;

	switch (key) {
		case GLUT_KEY_LEFT :
			anggle -= 0.01f;
			lx = sin(anggle);
			//lz = -cos(anggle);
			//ly = cos(anggle);
			break;
		case GLUT_KEY_RIGHT :
			anggle += 0.01f;
			lx = sin(anggle);
			//lz = -cos(anggle);
            //ly = cos(anggle);

			break;
		case GLUT_KEY_UP :
            anggle += 0.01f;
			ly = tan(anggle);
			break;
		case GLUT_KEY_DOWN :
			anggle -= 0.01f;
			ly = tan(anggle);
			break;


	}
}*/


void keyboard(unsigned char key, int x, int y)
{
	//float fraction = 0.1f;
// List all youe keyboard keys from assignment two her for body movement
	switch (key)
	{
case 'f':
		moveForward();
        break;
case 'b':
		moveBack();
        break;


/*case 'f':
		SetBound();
		if (eye[0]<0.25 && eye[0]>-0.25 && eye[2]<Zmax && eye[2]>Zmin)
			moveForward();
		else
			correct();
		break;
	case 'b':
		SetBound();
		if (eye[0]<0.25 && eye[0]>-0.25 && eye[2]<Zmax && eye[2]>Zmin)
			moveBack();
		else
			correct();

		break;
	case 27:
		exit(0);

		break;
	case ' ':
		if (DRot == 0 || DRot == 90)
		{
			if (DRot)
				DTimer1(0);
			else
				DTimer2(0);
		}
		break;*/
















       //case 'z' :
          // rotatePoint(0.0, 45.0, 2.0);
          // glutPostRedisplay();
        //fov+=10.0f;
       //x += lx * fraction;
       //z += lz * fraction;
       //y  += ly * fraction;
       //break;
//case 'Z' :
    //fov-=10.0f;
  //     x -= lx * fraction;
    //   z -= lz * fraction;
      // y  -= ly * fraction;
     //  break;
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


void changeSize(int w, int h)
{

// Prevent a divide by zero, when window is too short
// (you cant make a window of zero width).
if (h == 0)
h = 1;
float ratio = w * 1.0 / h;

// Use the Projection Matrix
glMatrixMode(GL_PROJECTION);

// Reset Matrix
glLoadIdentity();

// Set the viewport to be the entire window
glViewport(0, 0, w, h);

// Set the correct perspective.
gluPerspective(60.0f, ratio, 0.1f, 100.0f);

// Get Back to the Modelview
glMatrixMode(GL_MODELVIEW);
}



int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB| GLUT_DEPTH);
	glutInitWindowSize(1000, 1000);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("body");
	init();
	glutReshapeFunc(changeSize);
	glutDisplayFunc(display);
    glutSpecialFunc(specialKeys);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
    glutMotionFunc(motion);



    //glutSpecialFunc(processSpecialKeys);

	glutTimerFunc(0,Timer1,0);

	glutMainLoop();
	return 0;
}

