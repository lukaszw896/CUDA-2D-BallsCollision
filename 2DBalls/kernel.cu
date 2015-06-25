#include <windows.h>  // for MS Windows
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h
#include <Math.h>     // Needed for sin, cos
#define PI 3.14159265f
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#define N 20
#define G 0.0006f
#define S 0.95f
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm> 

// Global variables
char title[] = "Bouncing Ball (2D)";  // Windowed mode's title
int windowWidth = 1040; 	// Windowed mode's width 
int windowHeight = 880; 	// Windowed mode's height
int windowPosX = 50;  	// Windowed mode's top-left corner x
int windowPosY = 50;  	// Windowed mode's top-left corner y

GLfloat ballRadius = 0.02f;   // Radius of the bouncing ball
GLfloat ballXMax, ballXMin, ballYMax, ballYMin;

int ileiteracji = 0;
int ileTysiecy = 1;
float sumtime = 0;
int refreshMillis = 35;  	// Refresh period in milliseconds

float mojeKuleX[N];
float mojeKuleY[N];
float mojeSpeedX[N];
float mojeSpeedY[N];
__device__ bool  collisionMatrix[N * N];
__device__ float speedTableX_C[N];
__device__ float speedTableY_C[N];
__device__ float tmpSpeedTableX[N];
__device__ float tmpSpeedTableY[N];
cudaError_t initData(int n, float *mojeSpeedXCUDA, float *mojeSpeedYCUDA);

float getRandomXY()
{
	int c = rand() % 4;
	float r = -1.0f + (rand() / (float)RAND_MAX * 2.0f);
	r = r + (c * 0.000005f);
	return r;
}

float getRandomSpeed()
{
	int c = rand() % 2;
	float a = 0.1f;
	float r = ((rand() / (float)RAND_MAX * a));
	if (c == 1)
		r = -r;
	return r;
}

void init5kulek()
{
	int i = 0;
	for (i = 0; i < N; i++)
	{
		mojeKuleY[i] = (getRandomXY());// *0.5f;
		mojeKuleX[i] = (getRandomXY());// *0.5f;
		mojeSpeedX[i] = getRandomSpeed();
		mojeSpeedY[i] = 0.0f;
		//collisionMatrix[i] = 0;
	}
	initData(N, mojeSpeedX, mojeSpeedY);
}

// Projection clipping area
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;

/* Initialize OpenGL Graphics */
void initGL() {
	glClearColor(0.0, 0.0, 0.0, 1.0); // Set background (clear) color to black
}

cudaError_t CalculateWithCuda(float *mojeKuleXCUDA, float *mojeSpeedXCUDA, float *mojeKuleYCUDA, float *mojeSpeedYCUDA, float xmax, float xmin, float ymax, float ymin, int n, float grav, float spre, float radius);

bool detectCPUColision(float X1, float Y1, float X2, float Y2, float radius)
{
	//boolean colide = false;

	float xd = X1 - X2;
	float yd = Y1 - Y2;

	float sumRadius = radius + radius;
	float sqrRadius = sumRadius * sumRadius;

	float distSqr = (xd * xd) + (yd * yd);

	if (distSqr <= sqrRadius)
	{
		return true;
	}

	return false;
}

void addCPUKernel(float *mojeKuleXCUDA, float *mojeSpeedXCUDA, float *mojeKuleYCUDA, float *mojeSpeedYCUDA, float xmax, float xmin, float ymax, float ymin, int n, float grav, float spre, float radius, int *mojeDetectMem)
{
	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	//for(int k = 0 ; k < 2 ; k++)
	for (int i = 0; i < n; i++){
		if (true) {

			mojeKuleXCUDA[i] += mojeSpeedXCUDA[i];
			mojeKuleYCUDA[i] += mojeSpeedYCUDA[i];

			float pozX = mojeKuleXCUDA[i];
			float pozY = mojeKuleYCUDA[i];
			//int index = 0;
			if (mojeDetectMem[i] == 0){
				for (int p = 0; p < n; p++)
				{
					if (p != i && true == detectCPUColision(mojeKuleXCUDA[p], mojeKuleYCUDA[p], pozX, pozY, radius))
					{
						mojeDetectMem[i] == 7;
						if ((mojeSpeedXCUDA[i] == 0 && mojeSpeedXCUDA[p] != 0) || (mojeSpeedXCUDA[i] != 0 && mojeSpeedXCUDA[p] == 0) || (mojeSpeedXCUDA[i] > 0 && mojeSpeedXCUDA[p] < 0) || (mojeSpeedXCUDA[i] < 0 && mojeSpeedXCUDA[p]>0))
						{
							mojeSpeedXCUDA[i] = -mojeSpeedXCUDA[i];
							mojeKuleXCUDA[i] += mojeSpeedXCUDA[i];
							mojeSpeedXCUDA[i] = (0.8)*mojeSpeedXCUDA[i];
						}

						if ((mojeSpeedYCUDA[i] == 0 && mojeSpeedYCUDA[p] != 0) || (mojeSpeedYCUDA[i] != 0 && mojeSpeedYCUDA[p] == 0) || (mojeSpeedYCUDA[i]>0 && mojeSpeedYCUDA[p] < 0) || (mojeSpeedYCUDA[i] < 0 && mojeSpeedYCUDA[p]>0))
						{
							mojeSpeedYCUDA[i] = -mojeSpeedYCUDA[i];
							mojeKuleYCUDA[i] += mojeSpeedYCUDA[i];
							mojeSpeedYCUDA[i] = (0.8)*mojeSpeedYCUDA[i];
						}


						break;
					}
				}
			}
			else
			{
				mojeDetectMem[0]--;
				if (mojeDetectMem[0] < 0)
					mojeDetectMem[0] = 0;
			}


			if (mojeKuleXCUDA[i] > xmax) {
				mojeKuleXCUDA[i] = xmax;

				mojeSpeedXCUDA[i] -= ((0.6)*spre);
				mojeSpeedXCUDA[i] = -mojeSpeedXCUDA[i];
			}
			else if (mojeKuleXCUDA[i] < xmin) {
				mojeKuleXCUDA[i] = xmin;
				mojeSpeedXCUDA[i] += ((0.6)*spre);
				mojeSpeedXCUDA[i] = -mojeSpeedXCUDA[i];
			}

			mojeSpeedYCUDA[i] -= grav;

			if (mojeKuleYCUDA[i] > ymax) {
				mojeKuleYCUDA[i] = ymax;
				mojeSpeedYCUDA[i] = 0.0f;
			}
			else if (mojeKuleYCUDA[i]  < ymin) {
				mojeKuleYCUDA[i] = ymin;
				mojeSpeedYCUDA[i] += spre;
				mojeSpeedYCUDA[i] = -mojeSpeedYCUDA[i];
			}

		}
		//i += blockDim.x * gridDim.x;
	}

}

void CalculateWithCPU(float *mojeKuleXCUDA, float *mojeSpeedXCUDA, float *mojeKuleYCUDA, float *mojeSpeedYCUDA, float xmax, float xmin, float ymax, float ymin, int n, float grav, float spre, float radius, int *mojeDetectMem)
{
	addCPUKernel(mojeKuleXCUDA, mojeSpeedXCUDA, mojeKuleYCUDA, mojeSpeedYCUDA, xmax, xmin, ymax, ymin, n, G, S, ballRadius, mojeDetectMem);
}

__device__ int detectCollision(GLfloat x, GLfloat y, int ballNumber, int numberOfBalls, GLfloat * possitionTableX, GLfloat * possitionTableY, GLfloat ballRadius){
	int collisionBall = -1;
	int num = ballNumber;
	for (int i = 0; i < numberOfBalls; i++){
		if (i != ballNumber){
			/*local*/
			GLfloat secondBallX = possitionTableX[i];
			GLfloat secondBallY = possitionTableY[i];
			GLfloat firstBallX = possitionTableX[ballNumber];
			GLfloat firstBallY = possitionTableY[ballNumber];
			GLfloat leftSide = (2 * ballRadius)*(2 * ballRadius);
			GLfloat rightSide = ((firstBallX - secondBallX)*(firstBallX - secondBallX) + (firstBallY - secondBallY)*(firstBallY - secondBallY));
			/**/
			if (leftSide > rightSide)
			{
				//collisionBall = ballsMatrix[coordinateX + i][coordinateY + j];
				if (collisionMatrix[ballNumber + i*numberOfBalls] == false){
					collisionMatrix[ballNumber + i*numberOfBalls] = true;
					collisionMatrix[i + numberOfBalls*ballNumber] = true;
					return i;
				}
			}
			else{
				collisionMatrix[ballNumber + numberOfBalls * i] = false;
				collisionMatrix[i + numberOfBalls * ballNumber] = false;
			}
		}
	}
	return -1;
}

__global__ void initGpuData(int n, float* speedTableX, float *speedTableY){
	for (int i = 0; i < n; i++){
		speedTableX_C[i] = speedTableX[i];
		speedTableY_C[i] = speedTableY[i];
		tmpSpeedTableX[i] = speedTableX[i];
		tmpSpeedTableY[i] = speedTableY[i];
	}	
	for (int i = 0; i < N*N; i++){
		collisionMatrix[i] = false;
	}
}

__global__ void addKernel(float *possitionXTable_C, float *possitionTableY_C, float xmax, float xmin, float ymax, float ymin, int n, float grav, float springness, float radius)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	while (k< n){
		if (k < n) {
			//int coordinateX = 50 * ballTable[k * 2] + 50;
			//int coordinateY = 50 * ballTable[k * 2 + 1] + 50;
			//ballsMatrix[coordinateX][coordinateY] = -1;
			// Animation Control - compute the location for the next refresh
			possitionXTable_C[k] += speedTableX_C[k];
			possitionTableY_C[k] += speedTableY_C[k];

			// Check if the ball exceeds the edges
			if (possitionXTable_C[k] > xmax) {
				possitionXTable_C[k] = xmax;
				speedTableX_C[k] = -speedTableX_C[k];
				speedTableX_C[k] *= springness;
				tmpSpeedTableX[k] = -tmpSpeedTableX[k];
				tmpSpeedTableX[k ] *= springness;
			}
			else if (possitionXTable_C[k] < xmin) {
				possitionXTable_C[k] = xmin;
				speedTableX_C[k] = -speedTableX_C[k];
				speedTableX_C[k] *= springness;
				tmpSpeedTableX[k] = -tmpSpeedTableX[k];
				tmpSpeedTableX[k] *= springness;
			}
			if (possitionTableY_C[k] > ymax) {
				possitionTableY_C[k] = ymax;
				speedTableY_C[k] = -speedTableY_C[k];
				speedTableY_C[k] *= springness;
				tmpSpeedTableY[k] = -tmpSpeedTableY[k];
				tmpSpeedTableY[k] *= springness;
			}
			else if (possitionTableY_C[k]< ymin) {
				possitionTableY_C[k] = ymin;
				speedTableY_C[k] = -speedTableY_C[k];
				speedTableY_C[k] *= springness;
				tmpSpeedTableY[k] = -tmpSpeedTableY[k];
				tmpSpeedTableY[k] *= springness;
			}
			//coordinateX = 50 * ballTable[k * 2] + 50;
			//coordinateY = 50 * ballTable[k * 2 + 1] + 50;
			//ballsMatrix[coordinateX][coordinateY] = k;

			int ballDetected = detectCollision(possitionXTable_C[k], possitionTableY_C[k], k, n, possitionXTable_C,possitionTableY_C ,radius);

			if (ballDetected != -1){
			tmpSpeedTableX[k * 2] = speedTableX_C[ballDetected] * springness;
			tmpSpeedTableX[ballDetected * 2] = speedTableX_C[k] * springness;
			tmpSpeedTableY[k] = speedTableY_C[ballDetected] * springness;
			tmpSpeedTableY[ballDetected] = speedTableY_C[k] * springness;
			}

			//gravity
			speedTableY_C[k] -= 0.01f;
			tmpSpeedTableY[k] -= 0.01f;
		}
		k += blockDim.x * gridDim.x;
	}
	k = blockIdx.x * blockDim.x + threadIdx.x;
	while (k < n){
		if (k < n) {
			speedTableX_C[k] = tmpSpeedTableX[k];
			speedTableY_C[k] = tmpSpeedTableY[k];
		}
		k += blockDim.x * gridDim.x;
	}

}
cudaError_t initData(int n, float *mojeSpeedXCUDA, float *mojeSpeedYCUDA)
{
	float *dev_a = 0;
	float *dev_b = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, mojeSpeedXCUDA, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, mojeSpeedYCUDA, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	initGpuData << <1, 1 >> >(N, dev_a, dev_b);

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	/*cudaFree(dev_e);*/

	return cudaStatus;
}

cudaError_t CalculateWithCuda(float *mojeKuleXCUDA, float *mojeSpeedXCUDA, float *mojeKuleYCUDA, float *mojeSpeedYCUDA, float xmax, float xmin, float ymax, float ymin, int n, float grav, float spre, float radius, bool *collisionMatrix)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	float *dev_d = 0;
	bool *dev_e = 0;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)	.
	cudaStatus = cudaMalloc((void**)&dev_c, n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, mojeKuleXCUDA, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_c, mojeKuleYCUDA, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <10, 1000 >> >(dev_a,  dev_c,  xmax, xmin, ymax, ymin, n, G, S, ballRadius);

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
	cudaStatus = cudaMemcpy(mojeKuleXCUDA, dev_a, n * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(mojeKuleYCUDA, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	/*cudaFree(dev_e);*/

	return cudaStatus;
}


static double second(void)
{
	LARGE_INTEGER t;
	static double oofreq;
	static int checkedForHighResTimer;
	static BOOL hasHighResTimer;
	if (!checkedForHighResTimer) {
		hasHighResTimer = QueryPerformanceFrequency(&t);
		oofreq = 1.0 / (double)t.QuadPart;
		checkedForHighResTimer = 1;
	}
	if (hasHighResTimer) {
		QueryPerformanceCounter(&t);
		return (double)t.QuadPart * oofreq;
	}
	else {
		return (double)GetTickCount() / 1000.0;
	}
}
/* Callback handler for window re-paint event */
void display() {
	glClear(GL_COLOR_BUFFER_BIT);  // Clear the color buffer
	glMatrixMode(GL_MODELVIEW);	// To operate on the model-view matrix
	glLoadIdentity();          	// Reset model-view matrix


	//glTranslatef(ballX, ballY, 0.0f);  // Translate to (xPos, yPos)
	// Use triangular segments to form a circle
	int i = 0;
	for (i = 0; i < N; i++){

		glBegin(GL_TRIANGLE_FAN);
		glColor3f(2.0f, 0.0f, 2.0f);  // Blue

		glVertex2f(mojeKuleX[i], mojeKuleY[i]);   	// Center of circle // tutaj coordynaty
		int numSegments = 12;
		GLfloat angle;
		for (int j = 0; j <= numSegments; j++) { // Last vertex same as first vertex
			angle = j * 2.0f * PI / numSegments;  // 360 deg for all segments
			glVertex2f((cos(angle) * ballRadius) + mojeKuleX[i], (sin(angle) * ballRadius) + mojeKuleY[i]); // + coordynaty
		}
		glEnd();
	}
	glutSwapBuffers();  // Swap front and back buffers (of double buffered mode)

	// Animation Control - compute the location for the next refresh

	ileiteracji++;
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaError_t cudaStatus = CalculateWithCuda(mojeKuleX, mojeSpeedX, mojeKuleY, mojeSpeedY, ballXMax, ballXMin, ballYMax, ballYMin, N, G, S, ballRadius, collisionMatrix);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	sumtime += time;
	// 
	if (ileiteracji % 100 == 0)
	{
		printf("Average fps %d iteration is: %f \n", (ileTysiecy * 100), (float)(1000 / (sumtime / 100)));
		ileiteracji = 0;
		ileTysiecy++;
		sumtime = 0;
	}
	/*
	ileiteracji++;
	float hostTime;
	double startTime, stopTime, elapsed;
	startTime = second();
	CalculateWithCPU(mojeKuleX, mojeSpeedX, mojeKuleY, mojeSpeedY, ballXMax, ballXMin, ballYMax, ballYMin, N, G, S, ballRadius, mojeDetectMem);

	stopTime = second();
	hostTime = (stopTime - startTime) * 1000;

	sumtime += hostTime;
	if (ileiteracji % 100 == 0 )
	{
	printf("Average fps %d iteration is: %f \n", (ileTysiecy * 100), (float)(1000/(sumtime / 100)));
	ileiteracji = 0;
	ileTysiecy++;
	sumtime = 0;
	}*/


}

/* Call back when the windows is re-sized */
void reshape(GLsizei width, GLsizei height) {
	// Compute aspect ratio of the new window
	if (height == 0) height = 1;            	// To prevent divide by 0
	GLfloat aspect = (GLfloat)width / (GLfloat)height;

	// Set the viewport to cover the new window
	glViewport(0, 0, width, height);

	// Set the aspect ratio of the clipping area to match the viewport
	glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
	glLoadIdentity();         	// Reset the projection matrix
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
	glutPostRedisplay();	// Post a paint request to activate display()
	glutTimerFunc(refreshMillis, Timer, 0); // subsequent timer call at milliseconds
}

/* Main function: GLUT runs as a console application starting at main() */
int main(int argc, char** argv) {

	int IleMialyFps[5];
	int IlejestKulek[5] = { 1000, 2000, 4000, 7000, 10000 };

	srand(time(NULL));
	glutInit(&argc, argv);        	// Initialize GLUT
	glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
	glutInitWindowSize(windowWidth, windowHeight);  // Initial window width and height
	glutInitWindowPosition(windowPosX, windowPosY); // Initial window top-left corner (x, y)
	glutCreateWindow(title);  	// Create window with given title



	init5kulek();
	glutDisplayFunc(display); 	// Register callback handler for window re-paint
	glutReshapeFunc(reshape); 	// Register callback handler for window re-shape
	glutTimerFunc(0, Timer, 0);   // First timer call immediately
	initGL();                 	// Our own OpenGL initialization
	glutMainLoop();           	// Enter event-processing loop
	return 0;
}