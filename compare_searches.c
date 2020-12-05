// much of this was cribbed from 
// https://algorithmica.org/en/eytzinger
// https://stackoverflow.com/questions/3893937/sorting-an-array-in-c

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdalign.h>
#include <time.h>

int eytzinger_helper(int *input, int *output, int n, int i, int k) {
    if (k <= n) {
        i = eytzinger_helper(input, output, n, i, 2 * k);
        output[k] = input[i++];
        i = eytzinger_helper(input, output, n, i, 2 * k + 1);
    }
    return i;
}

void eytzinger(int *input, int *output, int n) {
	eytzinger_helper(input, output, n, 0, 1);
}

int compare( const void* a, const void* b)
{
     int int_a = * ( (int*) a );
     int int_b = * ( (int*) b );

     if ( int_a == int_b ) return 0;
     else if ( int_a < int_b ) return -1;
     else return 1;
}


int eytzinger_search(int* array, int n, int x) {
	int block_size = 16;
    int k = 1;
    while (k <= n) {
        __builtin_prefetch(array + k * block_size);
        k = 2 * k + (array[k] < x);
    }
    k >>= __builtin_ffs(~k);
    return k;
}

int binary_search(int* array, int n, int x) {
    int l = 0, r = n - 1;
    while (l < r) {
        int t = (l + r) / 2;
        if (array[t] >= x)
            r = t;
        else
            l = t + 1;
    }
    return array[l];
}

int main() {
	const int n = 1 << 20; // ~1e6
	alignas(64) int *input = (int*)calloc(n, sizeof(int));
	for (int i = 0; i < n; i++) {
		input[i] = random();
	}

	alignas(64) int *output = (int*)calloc(n+1, sizeof(int));
	qsort(input, n, sizeof(int), compare );
	eytzinger(input, output, n);

	clock_t start, end;
	double time_taken;
	int x;

	start = clock();
	for (int target = 0; target < 100000; target++) {
		x = eytzinger_search(output, n, input[target]);
	}
	end = clock();
	time_taken = (double)(end - start) / (double)(CLOCKS_PER_SEC);

	printf("eytzinger: %.2f msec, x=%d\n", time_taken * 1000, x);

	start = clock();
	for (int target = 0; target < 100000; target++) {
		x = binary_search(input, n, output[target]);
	}
	end = clock();
	time_taken = (double)(end - start) / (double)(CLOCKS_PER_SEC);

	printf("binary: %.2f msec, x=%d\n", time_taken * 1000, x);

	start = clock();
	for (int target = 0; target < 100000; target++) {
		x = eytzinger_search(output, n, input[target]);
	}
	end = clock();
	time_taken = (double)(end - start) / (double)(CLOCKS_PER_SEC);

	printf("eytzinger: %.2f msec, x=%d\n", time_taken * 1000, x);

	free(input);
	free(output);

	return 0;
}