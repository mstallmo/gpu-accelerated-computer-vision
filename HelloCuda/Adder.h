#pragma once
class Adder
{
public:
    Adder(int const firstValue, int const secondValue);
    ~Adder();

    int const add();

private:
    int const firstValue;
    int const secondValue;
};

