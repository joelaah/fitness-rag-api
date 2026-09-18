-- ========================================================================
-- Supabase Schema for Fitness App + RAG Integration
-- Includes Tables, Foreign Keys, Indexes, and Row Level Security (RLS)
-- ========================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------------------
-- 1. Profiles Table
-- ------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name TEXT,
    experience_level TEXT,
    preferred_training_goal TEXT,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- ------------------------------------------------------------------------
-- 2. Workout Sessions Table
-- ------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.workout_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    routine_id TEXT,
    routine_name TEXT,
    day_id TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    total_volume NUMERIC,
    total_reps INTEGER,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

ALTER TABLE public.workout_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own workout sessions"
    ON public.workout_sessions FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own workout sessions"
    ON public.workout_sessions FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own workout sessions"
    ON public.workout_sessions FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own workout sessions"
    ON public.workout_sessions FOR DELETE
    USING (auth.uid() = user_id);

CREATE INDEX IF NOT EXISTS idx_workout_sessions_user_date 
    ON public.workout_sessions(user_id, started_at DESC);

-- ------------------------------------------------------------------------
-- 3. Workout Exercises Table
-- ------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.workout_exercises (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workout_session_id UUID NOT NULL REFERENCES public.workout_sessions(id) ON DELETE CASCADE,
    exercise_id TEXT,
    exercise_name TEXT,
    order_index INTEGER DEFAULT 0,
    primary_muscle_group TEXT,
    secondary_muscle_groups TEXT[] DEFAULT '{}',
    is_skipped BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

ALTER TABLE public.workout_exercises ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own workout exercises"
    ON public.workout_exercises FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.workout_sessions s
            WHERE s.id = workout_exercises.workout_session_id
            AND s.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert own workout exercises"
    ON public.workout_exercises FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.workout_sessions s
            WHERE s.id = workout_exercises.workout_session_id
            AND s.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update own workout exercises"
    ON public.workout_exercises FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM public.workout_sessions s
            WHERE s.id = workout_exercises.workout_session_id
            AND s.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete own workout exercises"
    ON public.workout_exercises FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.workout_sessions s
            WHERE s.id = workout_exercises.workout_session_id
            AND s.user_id = auth.uid()
        )
    );

CREATE INDEX IF NOT EXISTS idx_workout_exercises_session 
    ON public.workout_exercises(workout_session_id);

-- ------------------------------------------------------------------------
-- 4. Workout Sets Table
-- ------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.workout_sets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workout_exercise_id UUID NOT NULL REFERENCES public.workout_exercises(id) ON DELETE CASCADE,
    set_number INTEGER NOT NULL,
    target_reps INTEGER,
    completed_reps INTEGER,
    target_weight NUMERIC,
    completed_weight NUMERIC,
    duration_seconds INTEGER,
    distance NUMERIC,
    is_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

ALTER TABLE public.workout_sets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own workout sets"
    ON public.workout_sets FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.workout_exercises e
            JOIN public.workout_sessions s ON s.id = e.workout_session_id
            WHERE e.id = workout_sets.workout_exercise_id
            AND s.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert own workout sets"
    ON public.workout_sets FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.workout_exercises e
            JOIN public.workout_sessions s ON s.id = e.workout_session_id
            WHERE e.id = workout_sets.workout_exercise_id
            AND s.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update own workout sets"
    ON public.workout_sets FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM public.workout_exercises e
            JOIN public.workout_sessions s ON s.id = e.workout_session_id
            WHERE e.id = workout_sets.workout_exercise_id
            AND s.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete own workout sets"
    ON public.workout_sets FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.workout_exercises e
            JOIN public.workout_sessions s ON s.id = e.workout_session_id
            WHERE e.id = workout_sets.workout_exercise_id
            AND s.user_id = auth.uid()
        )
    );

CREATE INDEX IF NOT EXISTS idx_workout_sets_exercise 
    ON public.workout_sets(workout_exercise_id);

-- ------------------------------------------------------------------------
-- 5. AI Recommendations Table
-- ------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.ai_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    generated_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    analysis_start_date TIMESTAMPTZ,
    analysis_end_date TIMESTAMPTZ,
    sessions_analyzed INTEGER NOT NULL DEFAULT 0,
    recommendation_json JSONB NOT NULL,
    summary TEXT,
    model_provider TEXT DEFAULT 'gemini-3.6-flash + cohere-rerank',
    prompt_version TEXT DEFAULT 'v1',
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

ALTER TABLE public.ai_recommendations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own AI recommendations"
    ON public.ai_recommendations FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own AI recommendations"
    ON public.ai_recommendations FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE INDEX IF NOT EXISTS idx_ai_recommendations_user_date 
    ON public.ai_recommendations(user_id, generated_at DESC);
