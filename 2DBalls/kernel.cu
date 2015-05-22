
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>  // for MS Windows
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h
#include <Math.h>     // Needed for sin, cos
#include <list>
#define PI 3.14159265f

double fRand(double fMin, double fMax);
__device__ int detectCollision(GLfloat x, GLfloat y, int ballNumber, int numberOfBalls, GLfloat * ballTable, bool * colissionsMatrix,GLfloat ballRadius);
__global__ void calculatePosition_kernel(int *numberOfBalls, GLfloat * ballTable, GLfloat * speedTable, GLfloat * tmpSpeedTable, bool * colissionsMatrix, GLfloat *springness, GLfloat *ballXMax, GLfloat *ballXMin, GLfloat *ballYMax, GLfloat *ballYMin, GLfloat* ballRadius);
// Global variables
char title[] = "Bouncing Ball (2D)";  // Windowed mode's title
int windowWidth = 1000;     // Windowed mode's width
int windowHeight = 1000;     // Windowed mode's height
int windowPosX = 50;      // Windowed mode's top-left corner x
int windowPosY = 50;      // Windowed mode's top-left corner y

int  numberOfBalls = 12;

bool * colissionsMatrix;

int ballsMatrix[100][100];

GLfloat springiness = 0.9;

GLfloat * ballTable;
GLfloat * speedTable;

GLfloat * tmpBallTable;
GLfloat * tmpSpeedTable;

GLfloat ballRadius = 0.001f;   // Radius of the bouncing ball
GLfloat ballX = 0.0f;         // Ball's center (x, y) position
GLfloat ballY = 0.0f;
GLfloat ballXMax, ballXMin, ballYMax, ballYMin; // Ball's center (x, y) bounds
GLfloat xSpeed = 0.0002f;      // Ball's speed in x and y directions
GLfloat ySpeed = 0.00007f;
int refreshMillis = 30;      // Refresh period in milliseconds

// Projection clipping area
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;

/* Initialize OpenGL Graphics */
void initGL() {
	glClearColor(0.0, 0.0, 0.0, 1.0); // Set background (clear) color to black
}

/* Callback handler for window re-paint event */
void display() {
	glClear(GL_COLOR_BUFFER_BIT);  // Clear the color buffer
	glMatrixMode(GL_MODELVIEW);    // To operate on the model-view matrix
	glLoadIdentity();              // Reset model-view matrix
	float color = 0.0f;
	float color2 = 0.0f;
	
	for (int i = 0; i < numberOfBalls; i++){
		glPushMatrix();
		glTranslatef(ballTable[i*2], ballTable[i*2+1], 0.0f);  // Translate to (xPos, yPos)
		// Use triangular segments to form a circle
		glBegin(GL_TRIANGLE_FAN);
		glColor3f(0.1f, 0.5f, 1.0f);  // Blue
		color += 0.2f;
		color2 += 0.1f;
		glVertex2f(0.0f,0.0f);       // Center of circle
		int numSegments = 100;
		GLfloat angle;
		for (int i = 0; i <= numSegments; i++) { // Last vertex same as first vertex
			angle = i * 2.0f * PI / numSegments;  // 360 deg for all segments
			glVertex2f(cos(angle) * ballRadius, sin(angle) * ballRadius);
		}
		glEnd();
		glPopMatrix();
	}
		glutSwapBuffers();  // Swap front and back buffers (of double buffered mode)

		cudaError_t cudaStatus;

		int * numberOfballsCuda;
		GLfloat * ballTableCuda, * speedTableCuda, * tmpSpeedTableCuda;
		bool * colissionsMatrixCuda;
		GLfloat * springnessCuda , * ballRadiusCuda;
		GLfloat * ballXMaxCuda, * ballXMinCuda, * ballYMaxCuda, * ballYMinCuda;

		cudaStatus = cudaMalloc((void**)&numberOfballsCuda, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&ballTableCuda, numberOfBalls * sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&speedTableCuda, numberOfBalls * sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&tmpSpeedTableCuda, numberOfBalls * sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&colissionsMatrixCuda, numberOfBalls * numberOfBalls * sizeof(bool));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&springnessCuda, sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&ballXMaxCuda, sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&ballYMaxCuda, sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&ballXMinCuda, sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&ballYMinCuda, sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMalloc((void**)&ballRadiusCuda, sizeof(GLfloat));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}


		
		cudaStatus = cudaMemcpy(ballTableCuda, ballTable, sizeof(GLfloat) * numberOfBalls, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(numberOfballsCuda, &numberOfBalls, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(speedTableCuda, speedTable, sizeof(GLfloat) * numberOfBalls, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(tmpSpeedTableCuda, tmpSpeedTable, sizeof(GLfloat) * numberOfBalls, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(colissionsMatrixCuda, colissionsMatrix, sizeof(bool) * numberOfBalls *numberOfBalls, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(springnessCuda, &springiness, sizeof(GLfloat), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(ballXMaxCuda, &ballXMax, sizeof(GLfloat), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(ballYMaxCuda, &ballYMax, sizeof(GLfloat), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(ballXMinCuda, &ballXMin, sizeof(GLfloat), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(ballYMinCuda, &ballYMin, sizeof(GLfloat), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(ballRadiusCuda, &ballRadius, sizeof(GLfloat), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		calculatePosition_kernel << <1, 1 >> >(numberOfballsCuda, ballTableCuda, speedTableCuda, tmpSpeedTableCuda, colissionsMatrixCuda, springnessCuda, ballXMaxCuda, ballXMinCuda, ballYMaxCuda, ballYMinCuda, ballRadiusCuda);

		cudaStatus = cudaMemcpy(ballTable, ballTableCuda, sizeof(GLfloat) * numberOfBalls, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(speedTable, speedTableCuda, sizeof(GLfloat) * numberOfBalls, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(tmpSpeedTable, tmpSpeedTableCuda, sizeof(GLfloat) * numberOfBalls, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaMemcpy(colissionsMatrix, colissionsMatrixCuda, sizeof(bool) * numberOfBalls *numberOfBalls, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaFree(ballTableCuda);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaFree(speedTableCuda);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaFree(tmpSpeedTableCuda);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
		cudaStatus = cudaFree(colissionsMatrixCuda);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return;
		}
}

__global__ void calculatePosition_kernel(int *numberOfBalls, GLfloat * ballTable, GLfloat * speedTable, GLfloat * tmpSpeedTable, bool * colissionsMatrix, GLfloat *springness, GLfloat *ballXMax, GLfloat *ballXMin, GLfloat *ballYMax, GLfloat *ballYMin, GLfloat* ballRadius){

	int k = blockIdx.x * blockDim.x + threadIdx.x;

	while (k< *numberOfBalls){
		//int coordinateX = 50 * ballTable[k * 2] + 50;
		//int coordinateY = 50 * ballTable[k * 2 + 1] + 50;
		//ballsMatrix[coordinateX][coordinateY] = -1;
		// Animation Control - compute the location for the next refresh
		ballTable[k * 2] += speedTable[k * 2];
		ballTable[k * 2 + 1] += speedTable[k * 2 + 1];

		// Check if the ball exceeds the edges
		if (ballTable[k * 2] > *ballXMax) {
			ballTable[k * 2] = *ballXMax;
			speedTable[k * 2] = -speedTable[k * 2];
			speedTable[k * 2] *= *springness;
			tmpSpeedTable[k * 2] = -tmpSpeedTable[k * 2];
			tmpSpeedTable[k * 2] *= *springness;
		}
		else if (ballTable[k * 2] < *ballXMin) {
			ballTable[k * 2] = *ballXMin;
			speedTable[k * 2] = -speedTable[k * 2];
			speedTable[k * 2] *= *springness;
			tmpSpeedTable[k * 2] = -tmpSpeedTable[k * 2];
			tmpSpeedTable[k * 2] *= *springness;
		}
		if (ballTable[k * 2 + 1] > *ballYMax) {
			ballTable[k * 2 + 1] = *ballYMax;
			speedTable[k * 2 + 1] = -speedTable[k * 2 + 1];
			speedTable[k * 2 + 1] *= *springness;
			tmpSpeedTable[k * 2 + 1] = -speedTable[k * 2 + 1];
			tmpSpeedTable[k * 2 + 1] *= *springness;
		}
		else if (ballTable[k * 2 + 1]< *ballYMin) {
			ballTable[k * 2 + 1] = *ballYMin;
			speedTable[k * 2 + 1] = -speedTable[k * 2 + 1];
			speedTable[k * 2 + 1] *= *springness;
			tmpSpeedTable[k * 2 + 1] = -tmpSpeedTable[k * 2 + 1];
			tmpSpeedTable[k * 2 + 1] *= *springness;
		}
		//coordinateX = 50 * ballTable[k * 2] + 50;
		//coordinateY = 50 * ballTable[k * 2 + 1] + 50;
		//ballsMatrix[coordinateX][coordinateY] = k;

		if (ballTable[k * 2] > 1.0f || ballTable[k * 2] < -1.0f)
		{
			printf("sdf");
		}

		/*int ballDetected = detectCollision(ballTable[k * 2], ballTable[k * 2 + 1], k, *numberOfBalls, ballTable, colissionsMatrix,*ballRadius);

		if (ballDetected != -1){
			tmpSpeedTable[k * 2] = speedTable[ballDetected * 2] * *springness;
			tmpSpeedTable[ballDetected * 2] = speedTable[k * 2] * *springness;
			tmpSpeedTable[k * 2 + 1] = speedTable[ballDetected * 2 + 1] * *springness;
			tmpSpeedTable[ballDetected * 2 + 1] = speedTable[k * 2 + 1] * *springness;
		}*/

		//gravity
		speedTable[k * 2 + 1] -= 0.005;
		tmpSpeedTable[k * 2 + 1] -= 0.005;
	}

/*	while (k<*numberOfBalls){
		speedTable[k * 2] = tmpSpeedTable[k * 2];
		speedTable[k * 2 + 1] = tmpSpeedTable[k * 2 + 1];
	}*/
	
}

/* Call back when the windows is re-sized */
void reshape(GLsizei width, GLsizei height) {
	// Compute aspect ratio of the new window
	if (height == 0) height = 1;                // To prevent divide by 0
	GLfloat aspect = (GLfloat)width / (GLfloat)height;

	// Set the viewport to cover the new window
	glViewport(0, 0, width, height);

	// Set the aspect ratio of the clipping area to match the viewport
	glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
	glLoadIdentity();             // Reset the projection matrix
	if (width >= height) {
		clipAreaXLeft = -1.0 * aspect;
		clipAreaXRight = 1.0 * aspect;
		clipAreaYBottom = -1.0;
		clipAreaYTop = 1.0;
	}
	else {
		clipAreaXLeft = -1.0;
		clipAreaXRight = 1.0;
		clipAreaYBottom = -1.0 / aspect;
		clipAreaYTop = 1.0 / aspect;
	}
	gluOrtho2D(clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop);
	ballXMin = clipAreaXLeft + ballRadius;
	ballXMax = clipAreaXRight - ballRadius;
	ballYMin = clipAreaYBottom + ballRadius;
	ballYMax = clipAreaYTop - ballRadius;
}

/* Called back when the timer expired */
void Timer(int value) {
	glutPostRedisplay();    // Post a paint request to activate display()
	glutTimerFunc(refreshMillis, Timer, 0); // subsequent timer call at milliseconds
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;*/
	for (int i = 0; i < 100; i++){
		for (int j = 0; j < 100; j++){
			ballsMatrix[i][j] = -1;
		}
	}

	double tmpX, tmpY;

	
	ballTable = (GLfloat*)malloc(numberOfBalls);
	speedTable = (GLfloat*)malloc(numberOfBalls);
	tmpSpeedTable = (GLfloat*)malloc(numberOfBalls);
	int coordinateX, coordinateY;

	for (int i = 0; i < numberOfBalls; i++){
		
		tmpX = fRand(-1.0, 1.0);
		tmpY = fRand(-1.0, 1.0);
		ballTable[i * 2] = tmpX;
		ballTable[i * 2 + 1] = tmpY;

		coordinateX = 50 * tmpX + 50;
		coordinateY = 50 * tmpY + 50;

		ballsMatrix[coordinateX][coordinateY] = i;

		speedTable[i * 2] = 0.02f;
		speedTable[i * 2 + 1] = 0.007f;
		tmpSpeedTable[i * 2] = 0.02f;
		tmpSpeedTable[i * 2 + 1] = 0.007f;
	}

	
	colissionsMatrix = (bool *)malloc(numberOfBalls * numberOfBalls * sizeof(bool));

	for (int i = 0; i < numberOfBalls; i++){
		for (int j = 0; j < numberOfBalls; j++){
			colissionsMatrix[i+ j*numberOfBalls] = false;
		}
	}
	glutInit(&argc, argv);            // Initialize GLUT
	glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
	glutInitWindowSize(windowWidth, windowHeight);  // Initial window width and height
	glutInitWindowPosition(windowPosX, windowPosY); // Initial window top-left corner (x, y)
	glutCreateWindow(title);      // Create window with given title
	glutDisplayFunc(display);     // Register callback handler for window re-paint
	glutReshapeFunc(reshape);     // Register callback handler for window re-shape
	glutTimerFunc(0, Timer, 0);   // First timer call immediately
	initGL();                     // Our own OpenGL initialization
	glutMainLoop();               // Enter event-processing loop
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}
struct Point
{
	int x;
	int y;
};

__device__ int detectCollision(GLfloat x, GLfloat y,int ballNumber,int numberOfBalls, GLfloat * ballTable, bool * colissionsMatrix,GLfloat ballRadius){
	int collisionBall = -1;
	int num = ballNumber;
	for (int i = 0; i < numberOfBalls; i++){
		if (i != ballNumber){
			/*local*/
			GLfloat secondBallX = ballTable[i * 2];
			GLfloat secondBallY = ballTable[i * 2 + 1];
			GLfloat firstBallX = ballTable[ballNumber * 2];
			GLfloat firstBallY = ballTable[ballNumber * 2 + 1];
			GLfloat leftSide = (2 * ballRadius)*(2 * ballRadius);
			GLfloat rightSide = ((firstBallX - secondBallX)*(firstBallX - secondBallX) + (firstBallY - secondBallY)*(firstBallY - secondBallY));
			/**/
			if (leftSide > rightSide)
			{
				//collisionBall = ballsMatrix[coordinateX + i][coordinateY + j];
				if (colissionsMatrix[ballNumber + i*numberOfBalls] == false){
					colissionsMatrix[ballNumber+ i*numberOfBalls] = true;
					colissionsMatrix[i+ numberOfBalls*ballNumber] = true;
					return i;
				}
			}
			else{
				colissionsMatrix[ballNumber+ numberOfBalls * i] = false;
				colissionsMatrix[i + numberOfBalls * ballNumber] = false;
			}
		}
	}
	return -1;
}
