import React from 'react';
import { FlaskConical } from 'lucide-react';

/**
 * Honest label for screens that still render fixture data.
 *
 * Six patient/doctor screens import from `src/data/` instead of calling an API.
 * A judge who logs in as a new patient must be able to tell that apart from a
 * working backend, so every fixture-backed screen says so, in the UI, on the
 * screen itself — not only in the README.
 */
const FixtureNotice = ({ feature, source = 'src/data/dummyData.js', note }) => (
    <div className="mb-6 rounded-xl border-2 border-dashed border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 px-4 py-3 flex items-start gap-3">
        <FlaskConical className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="text-sm">
            <p className="font-semibold text-amber-900 dark:text-amber-200">Interactive preview · fixture data</p>
            <p className="text-amber-800 dark:text-amber-300">
                {note || `This ${feature} screen renders sample rows from ${source}. It is not connected to a backend route yet, and none of it is real patient data.`}
            </p>
        </div>
    </div>
);

export default FixtureNotice;
