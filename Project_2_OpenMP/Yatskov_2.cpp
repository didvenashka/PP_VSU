//  Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <omp.h>
#define arraySize 100
#define launchTimes 100
#define e 10000
#define blockWarp (arraySize * arraySize / 10)
#define columnWarp (arraySize / 20)
#define leftEdge 1 
#define rightEdge 10
#define h ((float)(rightEdge - leftEdge))/arraySize
using namespace std;

struct funcOut
{
	double duration;
	float result;
};

float function(float x, float y)
{
	return (sin(x) * sin(x) + cos(y) * cos(y)) * h * h;
}

float max(float x, float y)
{
	return x > y ? x : y;
}

float min(float x, float y)
{
	return x < y ? x : y;
}

void showProperties()
{
#ifdef  _OPENMP
	cout << "OpenMP enabled. With version: " << _OPENMP << endl;
	omp_set_nested(true);
	cout << "omp nested: " << omp_get_nested();
#else
	cout << "OpenMP disabled.";
#endif
	printf("\n\n");
	cout << "Size of array: " << arraySize << "x" << arraySize << endl;
	cout << "Block warp (for block module): " << blockWarp << endl;
	cout << "Number of launches: " << launchTimes << endl;
	cout << "Error in the output of results: " << 1.0 / e;
	printf("\n\n");
}

void resetClock()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	int c = 0;
	start_time_f = omp_get_wtime();
#pragma omp parallel
	{
#pragma omp for schedule(guided)
		for (int i = 0; i < arraySize; i++)
		{
			for (int j = 0; j < arraySize; j++)
			{
				c++;
			}
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
}

funcOut functionMainThread()
{
	double start_time, end_time, duration;
	float result = 0.0;
	float x, y;
	start_time = omp_get_wtime();
	for (int i = 0; i < arraySize; i++)
	{
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			result += function(x, y);
		}
	}
	end_time = omp_get_wtime();
	duration = end_time - start_time;
	funcOut out;
	out.duration = duration;
	out.result = result;
	//cout << "results get consistently  >>> " << result << endl << "for " << duration << " miliseconds.\n";
	return out;
}

funcOut functionOpenMP_column_guided()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();

#pragma omp parallel for schedule(guided) private(x, y) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}

	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP [row] [guided]   >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_column_static()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();

#pragma omp parallel for schedule(static) private(x, y) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}

	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [row] [static]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_column_dynamic()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();

#pragma omp parallel for schedule(dynamic) private(x,y) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [row] [dynamic]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_row_guided()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();
	for (int i = 0; i < arraySize; i++)
	{
#pragma omp parallel for schedule(guided, columnWarp) private(x,y) reduction(+:omp_res)
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}

	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [column] [guided]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_row_static()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();
	for (int i = 0; i < arraySize; i++)
	{
#pragma omp parallel for schedule(static, columnWarp) private(x, y) reduction(+:omp_res)
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [column] [static]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_row_dynamic()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();
	for (int i = 0; i < arraySize; i++)
	{
#pragma omp parallel for schedule(dynamic, columnWarp) private(x, y) reduction(+:omp_res)
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}

	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [column] [dynamic]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_block_guided()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	funcOut out;
	start_time_f = omp_get_wtime();
#pragma omp parallel for schedule (guided, blockWarp) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
#pragma omp parallel for schedule(guided, blockWarp) private(x, y) 
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}

	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [block] [guided]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_block_static()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	funcOut out;
	float x, y;
	start_time_f = omp_get_wtime();
	omp_set_nested(true);
#pragma omp parallel for schedule(static, blockWarp) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
#pragma omp parallel for schedule(static, blockWarp) private(x, y) 
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}

	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [block] [static]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_block_dynamic()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();
	int i, j;
#pragma omp parallel for schedule(dynamic, blockWarp) reduction(+:omp_res)
	for (i = 0; i < arraySize; i++)
	{
#pragma omp parallel for schedule(dynamic, blockWarp) private(x, y)
		for (j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP, [block] [dynamic]  >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_collapse_guided()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();
#pragma omp parallel for schedule(guided) private(x, y) collapse(2) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP [collapse] [guided]   >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_collapse_static()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();
#pragma omp parallel for schedule(static) private(x, y) collapse(2) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP [collapse] [static]   >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

funcOut functionOpenMP_collapse_dynamic()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	float x, y;
	start_time_f = omp_get_wtime();
#pragma omp parallel for schedule(dynamic) private(x, y) collapse(2) reduction(+:omp_res)
	for (int i = 0; i < arraySize; i++)
	{
		for (int j = 0; j < arraySize; j++)
		{
			x = leftEdge + h * i;
			y = leftEdge + h * j;
			omp_res += function(x, y);
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
	funcOut out;
	out.duration = duration_f;
	out.result = omp_res;
	//cout << "results get with OMP [collapse] [dynamic]   >>> " << omp_res << endl << "for " << duration_f << " miliseconds.\n\n";
	return out;
}

void timing(funcOut func(), const char* funcName)
{
	funcOut get;
	float curDuration;
	float fullDuration = 0;
	float maxDuration = 0;
	float minDuration = 10000;
	for (int k = 0; k < launchTimes; k++)
	{
		resetClock();
		get = func();
		curDuration = get.duration;
		fullDuration += curDuration;
		maxDuration = max(curDuration, maxDuration);
		minDuration = min(curDuration, minDuration);
	}

	printf("\n ");
	printf(funcName);
	printf("\tget result : %.2f\n", get.result);
	printf("timings in seconds: min %.4f; avg %.4f; max %.4f;", round(minDuration * e) / e, round((fullDuration / launchTimes) * e) / e, round(maxDuration * e) / e);
	printf("\n\n");
}

int main()
{
	showProperties();

	timing(functionMainThread, "MAIN THREAD");

	timing(functionOpenMP_column_static, "COLUMN STATIC");
	timing(functionOpenMP_column_dynamic, "COLUMN DYNAMIC");
	timing(functionOpenMP_column_guided, "COLUMN GUIDED");

	timing(functionOpenMP_row_static, "ROW STATIC");
	timing(functionOpenMP_row_dynamic, "ROW DYNAMIC");
	timing(functionOpenMP_row_guided, "ROW GUIDED");

	timing(functionOpenMP_block_static, "BLOCK STATIC");
	timing(functionOpenMP_block_dynamic, "BLOCK DYNAMIC");
	timing(functionOpenMP_block_guided, "BLOCK GUIDED");


	timing(functionOpenMP_collapse_static, "COLLAPSE STATIC");
	timing(functionOpenMP_collapse_dynamic, "COLLAPSE DYNAMIC");
	timing(functionOpenMP_collapse_guided, "COLLAPSE GUIDED");


}
