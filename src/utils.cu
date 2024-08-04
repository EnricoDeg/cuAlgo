#include "utils.h"

int div_ceil(int numerator, int denominator)
{

	std::div_t res = std::div(numerator, denominator);
	return res.rem ? (res.quot + 1) : res.quot;
}