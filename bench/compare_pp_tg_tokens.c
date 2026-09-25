#include <stdio.h>
#include <stdlib.h>
int main(void) {
    srand(1);
    puts("2");
    for (int i = 1; i < 262144; i++) printf("%d\n", rand() % 262144);
    return ferror(stdout) ? 1 : 0;
}
