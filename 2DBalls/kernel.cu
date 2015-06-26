#include <windows.h>  // for MS Windows
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h
#include <Math.h>     // Needed for sin, cos
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PI 3.14159265f
#define N 6000
#define GRAVITY 0.01
#define SPRINGINESS 0.95
#define RADIUS 0.005
bool isCalculatedOnGPU = true;
float colorStep = 1.0 / N;

GLfloat clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;
GLfloat ballXMax, ballXMin, ballYMax, ballYMin;

int drawCallsCount = 0;
int measureCount = 1;
float timeSum = 0;
int refreshMillis = 35;  	// Refresh period in milliseconds

float ballsCoordinateX[N];
float ballsCoordinateY[N];
float speedX[N];
float speedY[N];
bool  collisionMatrixCPU[N * N];

/*************   DEVICE   **************/
__device__ bool  collisionMatrix[N * N];
__device__ float speedTableX_C[N];
__device__ float speedTableY_C[N];


/*************FUNCTIONS*****************/
/***************************************/
//cudaError_t initData(float *mojeSpeedXCUDA, float *mojeSpeedYCUDA);
cudaError_t sendDataToGPU(int n, float *speedX, float *speedY);
cudaError_t CalculateWithCuda(float *ballTableX, float *ballTableY, float xmax, float xmin, float ymax, float ymin, bool *collisionMatrix);
__device__ int detectCollision(GLfloat x, GLfloat y, int ballNumber, GLfloat * possitionTableX, GLfloat * possitionTableY);
__global__ void calculateNewPositions(float *possitionXTable_C, float *possitionTableY_C, float xmax, float xmin, float ymax, float ymin);
__global__ void initGpuData(float* speedTableX, float *speedTableY);

float getRandomXY();
float getRandomSpeed();
void initData();

void Timer(int value);
void reshape(GLsizei width, GLsizei height);
void display();

static double second();
void  addCPUKernel(float *possitionXTable_C, float *possitionTableY_C, float *speedTableX, float *speedTableY, float xmax, float xmin, float ymax, float ymin);
int detectCPUCollision(GLfloat x, GLfloat y, int ballNumber, GLfloat * possitionTableX, GLfloat * possitionTableY);


/************   MAIN   *****************/
/***************************************/

int main(int argc, char** argv) {

	srand(time(NULL));
	glutInit(&argc, argv);        	// Initialize GLUT
	glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
	glutInitWindowSize(1000, 1000);  // Initial window width and height
	glutCreateWindow("2DBalls");  	// Create window with given title

	initData();
	glutDisplayFunc(display); 	// Register callback handler for window re-paint
	glutReshapeFunc(reshape); 	// Register callback handler for window re-shape
	glutTimerFunc(0, Timer, 0);   // First timer call immediately
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glutMainLoop();           	// Enter event-processing loop
	return 0;
}

/******  Functions Declaration  ********/
/***************************************/

cudaError_t sendDataToGPU(int n, float *speedX, float *speedY)
{
	float *dev_speedX = 0;
	float *dev_speedY = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_speedX, n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_speedY, n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_speedX, speedX, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_speedY, speedY, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	initGpuData << <1, 1 >> >(dev_speedX, dev_speedY);

Error:
	cudaFree(dev_speedX);
	cudaFree(dev_speedY);

	return cudaStatus;
}

cudaError_t CalculateWithCuda(float *ballTableX, float *ballTableY, float xmax, float xmin, float ymax, float ymin, bool *collisionMatrix)
{
	float *dev_ballTableX = 0;
	float *dev_ballTableY = 0;
	

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)	.
	cudaStatus = cudaMalloc((void**)&dev_ballTableX, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_ballTableY, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_ballTableX, ballTableX, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_ballTableY, ballTableY, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	calculateNewPositions << <10, 1000 >> >(dev_ballTableX, dev_ballTableY, xmax, xmin, ymax, ymin);

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
	cudaStatus = cudaMemcpy(ballTableX, dev_ballTableX, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(ballTableY, dev_ballTableY, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_ballTableX);
	cudaFree(dev_ballTableY);

	return cudaStatus;
}

__device__ int detectCollision(GLfloat x, GLfloat y, int ballNumber,GLfloat * possitionTableX, GLfloat * possitionTableY){
	int collisionBall = -1;
	int num = ballNumber;
	for (int i = 0; i < N; i++){
		if (i != ballNumber){
			/*local*/
			GLfloat secondBallX = possitionTableX[i];
			GLfloat secondBallY = possitionTableY[i];
			GLfloat firstBallX = possitionTableX[ballNumber];
			GLfloat firstBallY = possitionTableY[ballNumber];
			GLfloat leftSide = (2 * RADIUS)*(2 * RADIUS);
			GLfloat rightSide = ((firstBallX - secondBallX)*(firstBallX - secondBallX) + (firstBallY - secondBallY)*(firstBallY - secondBallY));
			/**/
			if (leftSide > rightSide)
			{
				//collisionBall = ballsMatrix[coordinateX + i][coordinateY + j];
				if (collisionMatrix[ballNumber + i*N] == false){
					collisionMatrix[ballNumber + i*N] = true;
					collisionMatrix[i + N*ballNumber] = true;
					return i;
				}
			}
			else{
				collisionMatrix[ballNumber + N * i] = false;
				collisionMatrix[i + N * ballNumber] = false;
			}
		}
	}
	return -1;
}

__global__ void calculateNewPositions(float *possitionXTable_C, float *possitionTableY_C, float xmax, float xmin, float ymax, float ymin)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	while (k< N){
		if (k < N) {

			possitionXTable_C[k] += speedTableX_C[k];
			possitionTableY_C[k] += speedTableY_C[k];

			// Check if the ball exceeds the edges
			if (possitionXTable_C[k] > xmax) {
				possitionXTable_C[k] = xmax;
				speedTableX_C[k] = -speedTableX_C[k];
				speedTableX_C[k] *= SPRINGINESS;
			}
			else if (possitionXTable_C[k] < xmin) {
				possitionXTable_C[k] = xmin;
				speedTableX_C[k] = -speedTableX_C[k];
				speedTableX_C[k] *= SPRINGINESS;
			}
			if (possitionTableY_C[k] > ymax) {
				possitionTableY_C[k] = ymax;
				speedTableY_C[k] = -speedTableY_C[k];
				speedTableY_C[k] *= SPRINGINESS;
			}
			else if (possitionTableY_C[k]< ymin) {
				possitionTableY_C[k] = ymin;
				speedTableY_C[k] = -speedTableY_C[k];
				speedTableY_C[k] *= SPRINGINESS;
			}

			int ballDetected = detectCollision(possitionXTable_C[k], possitionTableY_C[k], k,  possitionXTable_C, possitionTableY_C);

			if (ballDetected != -1){
				float tmpSpeedX = speedTableX_C[k] * SPRINGINESS;
				float tmpSpeedY = speedTableY_C[k] * SPRINGINESS;
				speedTableX_C[k] = speedTableX_C[ballDetected];
				speedTableY_C[k] = speedTableY_C[ballDetected];
				speedTableX_C[ballDetected] = tmpSpeedX;
				speedTableY_C[ballDetected] = tmpSpeedY;
			}

			//gravity
			speedTableY_C[k] -= GRAVITY;

		}
		k += blockDim.x * gridDim.x;
	}

}
__global__ void initGpuData(float* speedTableX, float *speedTableY){
	for (int i = 0; i < N; i++){
		speedTableX_C[i] = speedTableX[i];
		speedTableY_C[i] = speedTableY[i];
		/*tmpSpeedTableX[i] = speedTableX[i];
		tmpSpeedTableY[i] = speedTableY[i];*/
	}
	for (int i = 0; i < N*N; i++){
		collisionMatrix[i] = false;
	}
}



/* Called back when the timer expired */
void Timer(int value) {
	glutPostRedisplay();	// Post a paint request to activate display()
	glutTimerFunc(refreshMillis, Timer, 0); // subsequent timer call at milliseconds
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
	ballXMin = clipAreaXLeft + RADIUS;
	ballXMax = clipAreaXRight - RADIUS;
	ballYMin = clipAreaYBottom + RADIUS;
	ballYMax = clipAreaYTop - RADIUS;
}
void display() {
	glClear(GL_COLOR_BUFFER_BIT);  // Clear the color buffer
	glMatrixMode(GL_MODELVIEW);	// To operate on the model-view matrix
	glLoadIdentity();          	// Reset model-view matrix


	//glTranslatef(ballX, ballY, 0.0f);  // Translate to (xPos, yPos)
	// Use triangular segments to form a circle
	int i = 0;
	for (i = 0; i < N; i++){

		glBegin(GL_TRIANGLE_FAN);
		glColor3f(1.0f, colorStep*i, 0.0f);  // Blue

		glVertex2f(ballsCoordinateX[i], ballsCoordinateY[i]);   	// Center of circle // tutaj coordynaty
		int numSegments = 12;
		GLfloat angle;
		for (int j = 0; j <= numSegments; j++) { // Last vertex same as first vertex
			angle = j * 2.0f * PI / numSegments;  // 360 deg for all segments
			glVertex2f((cos(angle) * RADIUS) + ballsCoordinateX[i], (sin(angle) * RADIUS) + ballsCoordinateY[i]); // + coordynaty
		}
		glEnd();
	}
	glutSwapBuffers();  // Swap front and back buffers (of double buffered mode)

	if (isCalculatedOnGPU){
		drawCallsCount++;
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		cudaError_t cudaStatus = CalculateWithCuda(ballsCoordinateX,  ballsCoordinateY,  ballXMax, ballXMin, ballYMax, ballYMin, collisionMatrix);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&time, start, stop);
		timeSum += time;
		// 
		if (drawCallsCount == 100)
		{
			printf("Average fps %f . Draw calls count: %d \n", (float)(1000 / (timeSum / 100)),(measureCount * 100));
			drawCallsCount = 0;
			measureCount++;
			timeSum = 0;
		}
	}
	else{
		drawCallsCount++;
		float hostTime;
		double startTime, stopTime, elapsed;
		startTime = second();
		addCPUKernel(ballsCoordinateX, speedX, ballsCoordinateY, speedY, ballXMax, ballXMin, ballYMax, ballYMin);
		stopTime = second();
		hostTime = (stopTime - startTime) * 1000;
		timeSum += hostTime;
		if (drawCallsCount == 100)
		{
			printf("Average fps %f . Draw calls count: %d \n", (float)(1000 / (timeSum / 100)), (measureCount * 100));
			drawCallsCount = 0;
			measureCount++;
			timeSum = 0;
		}
	}
}

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

void initData()
{
	int i = 0;
	for (i = 0; i < N; i++)
	{
		ballsCoordinateX[i] = (getRandomXY());// *0.5f;
		ballsCoordinateY[i] = (getRandomXY());// *0.5f;
		speedX[i] = getRandomSpeed();
		speedY[i] = 0.0f;
	}
	for (int i = 0; i < N*N; i++){
		collisionMatrixCPU[i] = false;
	}
	sendDataToGPU(N, speedX, speedY);
}

double second()
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

void  addCPUKernel(float *possitionXTable_C, float *speedTableX, float *possitionTableY_C, float *speedTableY, float xmax, float xmin, float ymax, float ymin)
{
	
	for(int k=0;k<N;k++){

			possitionXTable_C[k] += speedTableX[k];
			possitionTableY_C[k] += speedTableY[k];

			if (possitionXTable_C[k] > xmax) {
				possitionXTable_C[k] = xmax;
				speedTableX[k] = -speedTableX[k];
				speedTableX[k] *= SPRINGINESS;
			}
			else if (possitionXTable_C[k] < xmin) {
				possitionXTable_C[k] = xmin;
				speedTableX[k] = -speedTableX[k];
				speedTableX[k] *= SPRINGINESS;
			}
			if (possitionTableY_C[k] > ymax) {
				possitionTableY_C[k] = ymax;
				speedTableY[k] = -speedTableY[k];
				speedTableY[k] *= SPRINGINESS;
			}
			else if (possitionTableY_C[k]< ymin) {
				possitionTableY_C[k] = ymin;
				speedTableY[k] = -speedTableY[k];
				speedTableY[k] *= SPRINGINESS;
			}

			int ballDetected = detectCPUCollision(possitionXTable_C[k], possitionTableY_C[k], k, possitionXTable_C, possitionTableY_C);

			if (ballDetected != -1){
				float tmpSpeedX = speedTableX[k] * SPRINGINESS;
				float tmpSpeedY = speedTableY[k] * SPRINGINESS;
				speedTableX[k] = speedTableX[ballDetected];
				speedTableY[k] = speedTableY[ballDetected];
				speedTableX[ballDetected] = tmpSpeedX;
				speedTableY[ballDetected] = tmpSpeedY;
			}

			speedTableY[k] -= GRAVITY;
	}

}
int detectCPUCollision(GLfloat x, GLfloat y, int ballNumber, GLfloat * possitionTableX, GLfloat * possitionTableY){
	int collisionBall = -1;
	int num = ballNumber;
	for (int i = 0; i < N; i++){
		if (i != ballNumber){

			GLfloat secondBallX = possitionTableX[i];
			GLfloat secondBallY = possitionTableY[i];
			GLfloat firstBallX = possitionTableX[ballNumber];
			GLfloat firstBallY = possitionTableY[ballNumber];
			GLfloat leftSide = (2 * RADIUS)*(2 * RADIUS);
			GLfloat rightSide = ((firstBallX - secondBallX)*(firstBallX - secondBallX) + (firstBallY - secondBallY)*(firstBallY - secondBallY));

			if (leftSide > rightSide)
			{
				if (collisionMatrixCPU[ballNumber + i*N] == false){
					collisionMatrixCPU[ballNumber + i*N] = true;
					collisionMatrixCPU[i + N*ballNumber] = true;
					return i;
				}
			}
			else{
				collisionMatrixCPU[ballNumber + N * i] = false;
				collisionMatrixCPU[i + N * ballNumber] = false;
			}
		}
	}
	return -1;
}