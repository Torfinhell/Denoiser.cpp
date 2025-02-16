#pragma once
#include"tensors.h"
struct Layer{
    public:
    virtual void forward(){}
    virtual ~Layer(){}
};
struct ExampleLayer: public Layer{
    void forward() override;
    ~ExampleLayer(){}
};