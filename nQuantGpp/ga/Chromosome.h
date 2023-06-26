#pragma once
#include <memory>
#include <vector>

namespace nQuantGA
{
	template <class T>
	class Chromosome
	{
		public:
			virtual float getFitness() = 0;
			virtual shared_ptr<T> crossover(const T& mother, int numberOfCrossoverPoints, float crossoverProbability) = 0;
			virtual bool dominates(const T* right) = 0;
			virtual void mutation(int mutationSize, float mutationProbability) = 0;
			virtual vector<double> getObjectives() const = 0;
			virtual vector<double>& getConvertedObjectives() = 0;
			virtual void resizeConvertedObjectives(int numObj) = 0;
			virtual shared_ptr<T> makeNewFromPrototype() = 0;
	};
}
