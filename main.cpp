//
// Created by Andrey Aralov on 2/24/25.
//
#include <vector>
#include <complex>
#include <iostream>

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

using Complex = std::complex<float>;

auto sqr( auto a ) -> decltype( a*a )
{
    return a*a;
}

float fid(float alpha, Complex lamb, Complex xi)
{
    using namespace std;

    auto R = abs( xi );
    auto nu = arg( xi );
    Complex nui( 0.f, nu );

    auto M = 1 / sqrt( cosh( R ) );
// prevent division by zero when xi == 0
//    auto a = 1.f / sqrt( exp( nui ) * sinh( 2 * R ) );
    auto ab = 1.f / ( 2.f * cosh( R ) );
    auto b = sqrt( 0.5f * exp( nui ) * tanh( R ) );
    auto k = exp( -nui ) * tanh( R );

    auto r = lamb.real();
    auto c = lamb.imag();

    auto N = M * exp( sqr(alpha)*(-1.f + 0.5f*k - b*b) - 0.5f*sqr(abs(lamb)) + 0.5f*k*sqr(lamb));
    auto val = N*( exp(2.f*ab*sqr(alpha)) * sinh(alpha*(-r + lamb*k + 2.f*ab*lamb + Complex(0.f, c)))
            + exp(-2.f*ab*sqr(alpha)) * sinh(alpha*(-r + lamb*k - 2.f*ab*lamb + Complex(0.f, c)) ) );
    return sqr(abs(val)) / ( 1 - exp(-4.f * sqr(abs(alpha))) );
}


int main()
{
    float min = -0.5f;
    float max = 0.5f;
    int steps = 400;
    auto from_step = [&] ( int step ) -> float
    {
        return min + (float)step * (max - min) / (float)steps;
    };

    struct S
    {
        float maxVal = 0;
        Complex maxLamb, maxXi;

        S operator | ( const S& s )
        {
            return s.maxVal > maxVal ? s : *this;
        }
    };

    auto res = tbb::parallel_reduce( tbb::blocked_range( 0, steps ), S{}, [&]( tbb::blocked_range<int> r, S running )
    {
        for ( int step_x1 = r.begin(); step_x1 < r.end(); ++step_x1 )
        {
            for ( int step_x2 = 0; step_x2 < steps; ++step_x2 )
            {
                for ( int step_y1 = 0; step_y1 < steps; ++step_y1 )
                {
                    for ( int step_y2 = 0; step_y2 < steps; ++step_y2 )
                    {
                        Complex lamb( from_step( step_x1 ), from_step( step_y1 ) );
                        Complex xi( from_step( step_x2 ), from_step( step_y2 ) );
                        if ( auto val = fid( 2, lamb, xi ); val > running.maxVal )
                        {
                            running.maxVal = val;
                            running.maxLamb = lamb;
                            running.maxXi = xi;
                        }
                    }
                }
            }
        }

        return running;
    }, &S::operator| );

    std::cout << res.maxVal << '\n' << res.maxLamb << ' ' << res.maxXi;
}