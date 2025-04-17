#include <math.h>

#ifndef _DCOMPLEX
#define _DCOMPLEX
typedef struct DCOMPLEX
{
	double r, i;
} dcomplex;

inline dcomplex Cadd(dcomplex a, dcomplex b)
{
	dcomplex c;
	c.r = a.r + b.r;
	c.i = a.i + b.i;
	return c;
}

inline dcomplex Csub(dcomplex a, dcomplex b)
{
	dcomplex c;
	c.r = a.r - b.r;
	c.i = a.i - b.i;
	return c;
}

inline dcomplex Cmul(dcomplex a, dcomplex b)
{
	dcomplex c;
	c.r = a.r * b.r - a.i * b.i;
	c.i = a.i * b.r + a.r * b.i;
	return c;
}

inline dcomplex Complex(double re, double im)
{
	dcomplex c;
	c.r = re;
	c.i = im;
	return c;
}

inline dcomplex Conjg(dcomplex z)
{
	dcomplex c;
	c.r = z.r;
	c.i = -z.i;
	return c;
}

inline dcomplex Cdiv(dcomplex a, dcomplex b)
{
	dcomplex c;
	double r, den;
	if (fabs(b.r) >= fabs(b.i))
	{
		r = b.i / b.r;
		den = b.r + r * b.i;
		c.r = (a.r + r * a.i) / den;
		c.i = (a.i - r * a.r) / den;
	}
	else
	{
		r = b.r / b.i;
		den = b.i + r * b.r;
		c.r = (a.r * r + a.i) / den;
		c.i = (a.i * r - a.r) / den;
	}
	return c;
}

inline double Cabs(dcomplex z)
{
	double x, y, ans, temp;
	x = fabs(z.r);
	y = fabs(z.i);
	if (x == 0.0)
		ans = y;
	else if (y == 0.0)
		ans = x;
	else if (x > y)
	{
		temp = y / x;
		ans = x * sqrtf(1.0f + temp * temp);
	}
	else
	{
		temp = x / y;
		ans = y * sqrtf(1.0f + temp * temp);
	}
	return ans;
}

inline dcomplex Csqrt(dcomplex z)
{
	dcomplex c;
	double x, y, w, r;
	if ((z.r == 0.0) && (z.i == 0.0))
	{
		c.r = 0.0;
		c.i = 0.0;
		return c;
	}
	else
	{
		x = fabs(z.r);
		y = fabs(z.i);
		if (x >= y)
		{
			r = y / x;
			w = sqrt(x) * sqrtf(0.5f * (1.0f + sqrtf(1.0f + r * r)));
		}
		else
		{
			r = x / y;
			w = sqrt(y) * sqrtf(0.5f * (r + sqrtf(1.0f + r * r)));
		}
		if (z.r >= 0.0)
		{
			c.r = w;
			c.i = z.i / (2.0f * w);
		}
		else
		{
			c.i = (z.i >= 0) ? w : -w;
			c.r = z.i / (2.0f * c.i);
		}
		return c;
	}
}

inline dcomplex RCmul(double x, dcomplex a)
{
	dcomplex c;
	c.r = x * a.r;
	c.i = x * a.i;
	return c;
}

inline dcomplex RCdiv(double x, dcomplex a)
{
	dcomplex c;
	c.r = x / a.r;
	c.i = -x * a.i / (a.r * a.r + a.i * a.i);
	return c;
}

inline dcomplex Cexp(dcomplex a)
{
	dcomplex c;
	double expReal = exp(a.r);
	c.r = expReal * cos(a.i);
	c.i = expReal * sin(a.i);
	return c;
}

__device__ __inline__ dcomplex devCadd(dcomplex a, dcomplex b)
{
	dcomplex c;
	c.r = a.r + b.r;
	c.i = a.i + b.i;
	return c;
}

__device__ __inline__ dcomplex devCsub(dcomplex a, dcomplex b)
{
	dcomplex c;
	c.r = a.r - b.r;
	c.i = a.i - b.i;
	return c;
}

__device__ __inline__ dcomplex devCmul(dcomplex a, dcomplex b)
{
	dcomplex c;
	c.r = a.r * b.r - a.i * b.i;
	c.i = a.i * b.r + a.r * b.i;
	return c;
}

__device__ __inline__ dcomplex devComplex(double re, double im)
{
	dcomplex c;
	c.r = re;
	c.i = im;
	return c;
}

__device__ __inline__ dcomplex devConjg(dcomplex z)
{
	dcomplex c;
	c.r = z.r;
	c.i = -z.i;
	return c;
}

__device__ __inline__ dcomplex devCdiv(dcomplex a, dcomplex b)
{
	dcomplex c;
	double r, den;
	if (fabs(b.r) >= fabs(b.i))
	{
		r = b.i / b.r;
		den = b.r + r * b.i;
		c.r = (a.r + r * a.i) / den;
		c.i = (a.i - r * a.r) / den;
	}
	else
	{
		r = b.r / b.i;
		den = b.i + r * b.r;
		c.r = (a.r * r + a.i) / den;
		c.i = (a.i * r - a.r) / den;
	}
	return c;
}

__device__ __inline__ double devCabs(dcomplex z)
{
	double x, y, ans, temp;
	x = fabs(z.r);
	y = fabs(z.i);
	if (x == 0.0)
		ans = y;
	else if (y == 0.0)
		ans = x;
	else if (x > y)
	{
		temp = y / x;
		ans = x * sqrtf(1.0f + temp * temp);
	}
	else
	{
		temp = x / y;
		ans = y * sqrtf(1.0f + temp * temp);
	}
	return ans;
}

__device__ __inline__ dcomplex devCsqrt(dcomplex z)
{
	dcomplex c;
	double x, y, w, r;
	if ((z.r == 0.0) && (z.i == 0.0))
	{
		c.r = 0.0;
		c.i = 0.0;
		return c;
	}
	else
	{
		x = fabs(z.r);
		y = fabs(z.i);
		if (x >= y)
		{
			r = y / x;
			w = sqrt(x) * sqrtf(0.5f * (1.0f + sqrtf(1.0f + r * r)));
		}
		else
		{
			r = x / y;
			w = sqrt(y) * sqrtf(0.5f * (r + sqrtf(1.0f + r * r)));
		}
		if (z.r >= 0.0)
		{
			c.r = w;
			c.i = z.i / (2.0f * w);
		}
		else
		{
			c.i = (z.i >= 0) ? w : -w;
			c.r = z.i / (2.0f * c.i);
		}
		return c;
	}
}

__device__ __inline__ dcomplex devRCmul(double x, dcomplex a)
{
	dcomplex c;
	c.r = x * a.r;
	c.i = x * a.i;
	return c;
}

__device__ __inline__ dcomplex devRCdiv(double x, dcomplex a)
{
	dcomplex c;
	c.r = x / a.r;
	c.i = -x * a.i / (a.r * a.r + a.i * a.i);
	return c;
}

__device__ __inline__ dcomplex devCexp(dcomplex a)
{
	dcomplex c;
	double expReal = exp(a.r);
	c.r = expReal * cos(a.i);
	c.i = expReal * sin(a.i);
	return c;
}
#endif