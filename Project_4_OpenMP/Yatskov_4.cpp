// Yatskov_4.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <omp.h>
#define SIZE 75
#define runCount 111
#define sectionCount 4
#define	sectionLen (SIZE / sectionCount)

using namespace std;

void copy(int** from, int** to)
{
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			to[i][j] = from[i][j];
}

void print(int** matr)
{
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
			cout << matr[i][j] << " \t";
		cout << endl;
	}
}

void print_graph(int** matr)
{
	for (int i = 0; i < SIZE; i++)
		for (int j = i; j < SIZE; j++)
			if (matr[i][j])
				cout << "\n" << i + 1 << " - " << j + 1 << " : " << matr[i][j] << endl;
}

int min(int a, int b)
{
	return a < b ? a : b;
}
float min(float a, float b)
{
	return a < b ? a : b;
}
float max(float a, float b)
{
	return a > b ? a : b;
}

float floyd(int** matr)
{
	double start_time, end_time, duration;
	start_time = omp_get_wtime();

	for (int v = 0; v < SIZE; v++)
		for (int i = 0; i < SIZE; i++)
			for (int j = 0; j < SIZE; j++)
				if (matr[i][v] && matr[j][v])
					matr[i][j] = min(matr[i][j], (matr[i][v] + matr[j][v]));

	end_time = omp_get_wtime();
	duration = end_time - start_time;
	return duration;
}

float floyd_parallel(int** matr)
{
	double start_time, end_time, duration;
	start_time = omp_get_wtime();

#pragma omp parallel for schedule(guided)
	for (int v = 0; v < SIZE; v++)
		for (int i = 0; i < SIZE; i++)
			for (int j = 0; j < SIZE; j++)
				if (matr[i][v] && matr[j][v])
					matr[i][j] = min(matr[i][j], (matr[i][v] + matr[j][v]));

	end_time = omp_get_wtime();
	duration = end_time - start_time;
	return duration;
}

float floyd_parallel_section(int** matr)
{
	int sctn_starts[sectionCount];
	int sctn_ends[sectionCount];
	for (int k = 0; k < sectionCount; k++)
	{
		sctn_starts[k] = k * sectionLen;
		sctn_ends[k] = (k + 1) * sectionLen;
	}
	sctn_ends[sectionCount-1] = int(SIZE);

	double start_time, end_time, duration;
	start_time = omp_get_wtime();

#pragma omp parallel sections
	{
#pragma omp section
		{
			for (int v = sctn_starts[0]; v < sctn_ends[0]; v++)
				for (int i = 0; i < SIZE; i++)
					for (int j = 0; j < SIZE; j++)
						if (matr[i][v] && matr[j][v])
							matr[i][j] = min(matr[i][j], (matr[i][v] + matr[j][v]));
		}
#pragma omp section
		{
			for (int v = sctn_starts[1]; v < sctn_ends[1]; v++)
				for (int i = 0; i < SIZE; i++)
					for (int j = 0; j < SIZE; j++)
						if (matr[i][v] && matr[j][v])
							matr[i][j] = min(matr[i][j], (matr[i][v] + matr[j][v]));
		}
#pragma omp section
		{
			for (int v = sctn_starts[2]; v < sctn_ends[2]; v++)
				for (int i = 0; i < SIZE; i++)
					for (int j = 0; j < SIZE; j++)
						if (matr[i][v] && matr[j][v])
							matr[i][j] = min(matr[i][j], (matr[i][v] + matr[j][v]));
		}
#pragma omp section
		{
			for (int v = sctn_starts[3]; v < sctn_ends[3]; v++)
				for (int i = 0; i < SIZE; i++)
					for (int j = 0; j < SIZE; j++)
						if (matr[i][v] && matr[j][v])
							matr[i][j] = min(matr[i][j], (matr[i][v] + matr[j][v]));
		}
	}

	end_time = omp_get_wtime();
	duration = end_time - start_time;
	return duration;
}

void init(int** matr, string patern = "zeros", bool directed = false) // avaible paterns {zeros/ones/random/increasing} 
{
	if (patern == "zeros")
	{
		for (int i = 0; i < SIZE; i++)
			for (int j = 0; j < SIZE; j++)
				matr[i][j] = 0;
	}
	else if (patern == "ones")
	{
		for (int i = 0; i < SIZE; i++)
			for (int j = 0; j < SIZE; j++)
				matr[i][j] = 1;
	}
	else if (patern == "random")
	{
		for (int i = 0; i < SIZE; i++)
			for (int j = 0; j < SIZE; j++)
				matr[i][j] = rand() % 20 + 5;
	}
	else if (patern == "increasing")
	{
		for (int i = 0; i < SIZE; i++)
			for (int j = 0; j < SIZE; j++)
				matr[i][j] = i / 3 + j / 2 + 1;
	}
	else cout << "\n::: Wrond parametr <patern> in init func\n";
	if (directed)
	{
		for (int i = 0; i < SIZE; i++)
			matr[i][i] = NAN;
	}
	else
	{
		for (int i = 0; i < SIZE; i++)
		{
			matr[i][i] = NAN;
			for (int j = i; j < SIZE; j++)
				matr[j][i] = matr[i][j];
		}
	}
}

void resetClock()
{
	double start_time_f, end_time_f, duration_f;
	float omp_res = 0.0;
	int c = 0;
	start_time_f = omp_get_wtime();
#pragma omp parallel for schedule(guided)
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			c++;
		}
	}
	end_time_f = omp_get_wtime();
	duration_f = end_time_f - start_time_f;
}

void timing(float func(int**), int** matr, const char* funcName)
{
	float curDuration;
	float fullDuration = 0;
	float maxDuration = 0;
	float minDuration = 10000;
	for (int k = 0; k < runCount; k++)
	{
		int** matr_cpy;
		matr_cpy = new int* [SIZE];
		for (int k = 0; k < SIZE; k++)
			matr_cpy[k] = new int[SIZE];
		copy(matr, matr_cpy);
		resetClock();
		curDuration = func(matr_cpy);
		fullDuration += curDuration;
		maxDuration = max(curDuration, maxDuration);
		minDuration = min(curDuration, minDuration);
	}
	cout << endl << funcName << endl;
	cout << "min " << round(minDuration * 10000) / 10000;
	cout << " / avg " << round((fullDuration / runCount) * 10000) / 10000;
	cout << " / max " << round(maxDuration * 10000) / 10000 << endl << endl;

}

int main()
{

	int** a;
	int** b;
	int** c;
	a = new int* [SIZE];
	b = new int* [SIZE];
	c = new int* [SIZE];
	for (int k = 0; k < SIZE; k++)
	{
		a[k] = new int[SIZE];
		b[k] = new int[SIZE];
		c[k] = new int[SIZE];
	}

	init(a, "random", false);
	copy(a, b);
	copy(a, c);

	if (SIZE < 6)
	{
		cout << "start matrix : \n";
		print(b);
		cout << "\n---------------------------\n";
		floyd(b);
		cout << "\n\nafter non-parallel : \n";
		print(b);
		cout << "\n---------------------------\n";
		print_graph(b);
		cout << "\n---------------------------\n";
		floyd_parallel_section (a);
		cout << "\n\nafter parallel : \n";
		print(a);
		cout << "\n---------------------------\n";
		print_graph(a);
		cout << "\n---------------------------\n\n";
	}

	timing(floyd, c, "Non parallel");
	timing(floyd_parallel, c, "Parallel");
	timing(floyd_parallel_section, c, "Parallel-section");
}
