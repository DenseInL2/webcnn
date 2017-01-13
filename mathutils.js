/*
 Instantiates a pseudo-random number source with standard deviation of 1 and mean of 0
 using the Marsaglia polar method. I've implemented this as a class rather than a function only
 because this allows it to save state (the extra value it generates) in a way that is easy to read
 and inspect in a debugger.
 */
class GaussianRNG
{
	constructor()
	{
		this.nextValue = NaN;
	}

	getNextRandom()
	{
		if ( !Number.isNaN( this.nextValue ) )
		{
			const ret = this.nextValue;
			this.nextValue = NaN;
			return ret;
		}

		let u, v, s = 0;
		while ( s > 1 || s == 0 )
		{
			u = Math.random() * 2.0 - 1.0;
			v = Math.random() * 2.0 - 1.0;
			s = u * u + v * v;
		}

		const mult = Math.sqrt( -2.0 * Math.log( s ) / s );
		this.nextValue = v * mult;
		return u * mult;
	}
}
