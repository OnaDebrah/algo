'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { advisorApi } from '@/lib/api/advisor'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { RecommendationCard } from '@/components/advisor/RecommendationCard'
import { Sparkles } from 'lucide-react'
import type { UserProfile, StrategyRecommendation } from '@/types'

export default function AdvisorPage() {
  const [profile, setProfile] = useState<Partial<UserProfile>>({
    capital: 100000,
    riskTolerance: 'Medium',
    experience: 'Intermediate',
  })

  const [recommendations, setRecommendations] = useState<StrategyRecommendation[]>([])

  const recommendMutation = useMutation({
    mutationFn: (profile: UserProfile) => advisorApi.getRecommendations(profile),
    onSuccess: (data) => setRecommendations(data),
  })

  const handleSubmit = () => {
    if (profile.goals && profile.riskTolerance && profile.experience) {
      recommendMutation.mutate(profile as UserProfile)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Sparkles className="h-8 w-8 text-primary" />
          AI Strategy Advisor
        </h1>
        <p className="text-muted-foreground">
          Get personalized strategy recommendations based on your profile
        </p>
      </div>

      {!recommendations.length ? (
        <Card>
          <CardHeader>
            <CardTitle>Tell us about your trading goals</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <Label>Capital</Label>
              <Input
                type="number"
                value={profile.capital}
                onChange={(e) => setProfile({ ...profile, capital: Number(e.target.value) })}
              />
            </div>

            <div className="space-y-3">
              <Label>Risk Tolerance</Label>
              <Select
                value={profile.riskTolerance}
                onValueChange={(value) => setProfile({ ...profile, riskTolerance: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Very Low">Very Low</SelectItem>
                  <SelectItem value="Low">Low</SelectItem>
                  <SelectItem value="Medium">Medium</SelectItem>
                  <SelectItem value="High">High</SelectItem>
                  <SelectItem value="Very High">Very High</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <Label>Experience Level</Label>
              <Select
                value={profile.experience}
                onValueChange={(value) => setProfile({ ...profile, experience: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Beginner">Beginner</SelectItem>
                  <SelectItem value="Intermediate">Intermediate</SelectItem>
                  <SelectItem value="Advanced">Advanced</SelectItem>
                  <SelectItem value="Expert">Expert</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button onClick={handleSubmit} className="w-full" disabled={recommendMutation.isPending}>
              {recommendMutation.isPending ? 'Analyzing...' : 'Get Recommendations'}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Your Personalized Recommendations</h2>
            <Button variant="outline" onClick={() => setRecommendations([])}>
              Start Over
            </Button>
          </div>

          {recommendations.map((rec) => (
            <RecommendationCard key={rec.strategyKey} recommendation={rec} />
          ))}
        </div>
      )}
    </div>
  )
}
