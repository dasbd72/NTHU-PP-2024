#include <cassert>
#include <iostream>
#include <vector>

void send_count_left(int n, float *data, float target, int &count) {
    if (target < data[0]) {
        count = n;
        return;
    }
    int l = 0, r = n;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (data[m] >= target) {
            r = m;  // Look for the first element that is greater than or equal to target
        } else {
            l = m + 1;  // Move right if data[m] < target
        }
    }
    // 'l' now points to the first element that is greater than or equal to 'target'
    count = n - l;
}

void send_count_right(int n, float *data, float target, int &count) {
    if (target < data[0]) {
        count = 0;
        return;
    }
    int l = 0, r = n;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (data[m] <= target) {
            l = m + 1;  // Move right because data[m] is less than or equal to target
        } else {
            r = m;  // Move left because data[m] is greater than target
        }
    }
    // 'l' will point to the first element greater than 'target', so count is the number of elements <= target
    count = l;
}

int test_send_count_left() {
    struct Testcase {
        int n;
        float data[10];
        float target;
        int count;
    };
    std::vector<Testcase> testcases = {
        {6, {1, 2, 3, 4, 5, 6}, 3, 4},
        {6, {1, 2, 3, 3, 5, 6}, 3, 4},
        {6, {1, 2, 3, 3, 5, 6}, 5, 2},
        {6, {1, 2, 3, 3, 5, 6}, 0, 6},
    };
    int error = 0;
    for (auto &tc : testcases) {
        int count;
        send_count_left(tc.n, tc.data, tc.target, count);
        if (count != tc.count) {
            std::cerr << "test_send_count_left FAIL, got " << count << ", want " << tc.count << std::endl;
            error++;
        }
    }
    return error;
}

int test_send_count_right() {
    struct Testcase {
        int n;
        float data[10];
        float target;
        int count;
    };
    std::vector<Testcase> testcases = {
        {6, {1, 2, 3, 4, 5, 6}, 3, 3},
        {6, {1, 2, 3, 3, 5, 6}, 3, 4},
        {6, {1, 2, 3, 3, 5, 6}, 5, 5},
        {6, {1, 2, 3, 3, 5, 6}, 7, 6},
    };
    int error = 0;
    for (auto &tc : testcases) {
        int count;
        send_count_right(tc.n, tc.data, tc.target, count);
        if (count != tc.count) {
            std::cerr << "test_send_count_right FAIL, got " << count << ", want " << tc.count << std::endl;
            error++;
        }
    }
    return error;
}

int main() {
    int error = 0;
    error += test_send_count_left();
    error += test_send_count_right();
    if (error) {
        std::cerr << "FAIL: " << error << " errors" << std::endl;
    } else {
        std::cout << "PASS" << std::endl;
    }
    return 0;
}