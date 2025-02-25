//
// Created by Andrey Aralov on 2/24/25.
//
#include <complex>
#include <iostream>
#include <format>
#include <print>

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

#include <boost/program_options.hpp>

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


namespace po = boost::program_options;

int main( int ac, char* av[] )
{
    float min = -1.f;
    float max = 1.f;
    int steps = 200;
    float alpha = 2;

    try{
        po::options_description desc("");
        desc.add_options()
            ( "steps", po::value( &steps )->default_value( steps ), "Number of steps in the grid (along one axis)" )
            ( "min", po::value( &min )->default_value( min ), "Minimum value of grid" )
            ( "max", po::value( &max )->default_value( max ), "Maximum value of grid" )
            ( "alpha", po::value( &alpha )->default_value( alpha ), "Alpha" )
        ;
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if ( vm.count("help") )
        {
            std::cout << desc << "\n";
            return 0;
        }
    }
    catch ( std::exception&e )
    {
        std::cerr << e.what() << '\n';
        exit( -1 );
    }
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
                        if ( auto val = fid( alpha, lamb, xi ); val >= running.maxVal )
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

    std::print( "{:.4f}|{:.4f} {:.4f}|{:.4f} {:.4f}\n", res.maxVal, res.maxLamb.real(), res.maxLamb.imag(), res.maxXi.real(), res.maxXi.imag() );
}